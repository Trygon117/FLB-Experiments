import torch
import torch.nn as nn

class Layer_Block(nn.Module):
    def __init__(self, hidden_dim, expansion=2, num_heads=4):
        super(Layer_Block, self).__init__()

        # Each incoming token get a learnable identity projection and a normalization layer
        self.F = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.L = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.B = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))

        # Mixes the 3 streams (F, L, B) at each position
        self.stream_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Mixes tokens across the sequence length
        self.seq_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Normalize the output after each attention operation
        self.norm_stream = nn.LayerNorm(hidden_dim)
        self.norm_seq = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion, hidden_dim)
        )

    def forward(self, fwd, lat, bck, pos_emb=None):
        batch_size, seq_len, hidden_dim = fwd.shape

        # Project and normalize each stream
        F = self.F(fwd)  # (batch_size, seq_len, hidden_dim)
        L = self.L(lat)  # (batch_size, seq_len, hidden_dim)
        B = self.B(bck)  # (batch_size, seq_len, hidden_dim)

        ### IM MAKING AN ARCHITECTUAL DECISION HERE THAT *ONLY* THE *LATERAL* TOKEN IS GENERATING THE QUERY BASED ON ITS PREVIOUS STATE ###

        # Flatten batch and sequence so each position is treated as an independent item
        F_flat = F.view(batch_size * seq_len, 1, hidden_dim)  # Shape: (batch_size * seq_len, 1, hidden_dim)
        L_flat = L.view(batch_size * seq_len, 1, hidden_dim)  # Shape: (batch_size * seq_len, 1, hidden_dim)
        B_flat = B.view(batch_size * seq_len, 1, hidden_dim)  # Shape: (batch_size * seq_len, 1, hidden_dim)

        # Join the three streams side by side to make keys and values of length 3
        kv_streams = torch.cat([F_flat, L_flat, B_flat], dim=1)  # Shape: (batch_size * seq_len, 3, hidden_dim)

        ### Because of the scale (3 tokens) it might actually be more efficient to do this dot product math manually here ###

        # Lateral query asks questions of the 3 streams at its own position
        stream_out, _ = self.stream_attn(query=L_flat, key=kv_streams, value=kv_streams, is_causal=False, need_weights=False) # Shape: (batch_size * seq_len, 1, hidden_dim)

        # Restore original sequence shape and add residual connection from L
        mixed_lat = stream_out.view(batch_size, seq_len, hidden_dim)    # Shape: (batch_size, seq_len, hidden_dim)
        mixed_lat = self.norm_stream(L + mixed_lat)                     # Shape: (batch_size, seq_len, hidden_dim)

        ### After computing an update for each individual token, we do sequence wise attention ###

        # Inject position into queries and keys to preserve semantic values
        if pos_emb is not None:
            q = mixed_lat + pos_emb
            k = mixed_lat + pos_emb
        else:
            q = mixed_lat
            k = mixed_lat
        v = mixed_lat

        # Generate causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=q.device)

        # Run sequence attention across time
        seq_out, _ = self.seq_attn(query=q, key=k, value=v, attn_mask=causal_mask, is_causal=True, need_weights=False) # Shape: (batch_size, seq_len, hidden_dim)

        # Normalize the sequence attention output
        seq_out = self.norm_seq(seq_out) # Shape: (batch_size, seq_len, hidden_dim)

        # Apply feedforward network with residual connection
        update = self.ffn(mixed_lat + seq_out) # Shape: (batch_size, seq_len, hidden_dim)

        return update # Shape: (batch_size, seq_len, hidden_dim)

class FLB_Model(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_heads = 4, sweep_iters = 1, num_layers = 6, layer_iters = 1, window_size = 64):
        super(FLB_Model, self).__init__()
        # model parameters
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.window_size = window_size

        # iteration parameters
        self.num_layers = num_layers
        self.layer_iters = layer_iters
        self.sweep_iters = sweep_iters

        # Standard deviation calculated from the hidden size to initialize prediction_slots pre-normalized
        init_scale = 1.0 / (hidden_dim ** 0.5)

        # For initializing new "empty" layers
        self.new_layer_context = nn.Parameter(torch.randn(num_layers, 1, 1, hidden_dim) * init_scale)
        
        # model layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(window_size, hidden_dim) # Cannot predict more future tokens than are in the window
        self.layers = nn.ModuleList([Layer_Block(hidden_dim, expansion=2, num_heads=num_heads) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, layers_out=None):
        batch_size, seq_len = x.shape
        fwd_stream = self.embedding(x)

        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embedding(positions).unsqueeze(0)  # Shape: (1, seq_len, hidden_dim)

        x_stream, layers_out = self.run_iterations(fwd_stream, layers_out, pos_emb=pos_emb)

        # Normalize and pass final output through prediction head
        x_stream = self.final_norm(x_stream)
        output = self.output(x_stream)
        return output, layers_out

    def run_iterations(self, x_stream, layers_out=None, pos_emb=None):
        batch_size, seq_len, _ = x_stream.shape

        # If no previous memory exists, expand the learned initial states
        if layers_out is None:
            layers_out = [self.new_layer_context[n].expand(batch_size, seq_len, self.hidden_dim) for n in range(self.num_layers)]

        for s in range(self.sweep_iters): # Go through each Sweep Iteration
            for n in range(self.num_layers): # Go through each layer
                # Backward signal from the layer above
                if n + 1 < self.num_layers:
                    back = layers_out[n+1]
                else: 
                    back = torch.zeros_like(x_stream)
                # Forward signal from layer below
                if(n == 0):
                    forward = x_stream
                else:
                    forward = layers_out[n-1]
                for l in range(self.layer_iters): # Go through every layer iteration
                    # Pass in the current forward input, the iterated lateral input, and the current backward input
                    layers_out[n] = forward + self.layers[n](forward, layers_out[n], back, pos_emb=pos_emb)
                ### end layer iterations
            ### end layers
        ### end sweep iterations

        # Update the input to the next sweep with the output of the last layer
        output = layers_out[self.num_layers - 1]

        return output, layers_out

    def stream_model(self, prompt, layer_context, stride):
        output, layer_context = self(prompt, layer_context)
        next_layer_context = self.shift_memory(layer_context, stride)
        return output, next_layer_context

    def train_model(self, sequence, optimizer=None, criterion=None, layer_context=None, stride=1, on_window=None):
        batch_size, seq_len = sequence.shape
        needed_len = self.window_size + 1

        assert seq_len >= needed_len, (
            f"Sequence length ({seq_len}) is too short. "
            f"Must be at least {needed_len} tokens for window + next token target."
        )

        if layer_context is None:
            layer_context = [self.new_layer_context[n].expand(batch_size, self.window_size, self.hidden_dim) for n in range(self.num_layers)]

        if criterion is None:
            criterion = nn.CrossEntropyLoss(reduction='none')

        total_loss = 0.0
        window_count = 0

        # Slide across the sequence
        for start in range(0, seq_len - self.window_size, stride):
            if optimizer is not None:
                optimizer.zero_grad()

            input = sequence[:, start : start + self.window_size]
            targets = sequence[:, start + 1 : start + self.window_size + 1]

            outputs, layer_context = self.stream_model(input, layer_context, stride)

            raw_loss = criterion(
                outputs.reshape(-1, self.vocab_size),
                targets.reshape(-1)
            ).reshape(batch_size, -1)

            step_loss = raw_loss.mean()
            step_loss.backward()

            if on_window is not None:
                on_window({
                    'model': self,
                    'window_idx': window_count,
                    'step_loss': step_loss.item(),
                    'input': input,
                    'outputs': outputs,
                    'predictions': outputs,
                    'targets': targets,
                    'raw_loss': raw_loss.detach()
                })

            if optimizer is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += step_loss.item()
            window_count += 1

        avg_loss = total_loss / max(1, window_count)
        return avg_loss, layer_context
    
    @torch.no_grad()
    def generate(self, prompt, num_tokens, temperature=0.8, layer_context=None, on_step=None):
        self.eval()
        batch_size = prompt.shape[0]
        tokens = prompt.clone()
        generated_count = 0

        if layer_context is None:
            layer_context = [self.new_layer_context[n].expand(batch_size, self.window_size, self.hidden_dim) for n in range(self.num_layers)]

        while generated_count < num_tokens:
            if tokens.shape[1] < self.window_size:
                pad_len = self.window_size - tokens.shape[1]
                pad = torch.zeros(batch_size, pad_len, dtype=tokens.dtype, device=tokens.device)
                input = torch.cat([pad, tokens], dim=1)
            else:
                input = tokens[:, -self.window_size:]

            outputs, layer_context = self.stream_model(input, layer_context, stride=1)

            last_logits = outputs[:, -1, :]
            probs = torch.softmax(last_logits / temperature, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, sampled], dim=1)
            generated_count += 1

            if on_step is not None:
                on_step({
                    'new_tokens': sampled,
                    'tokens_so_far': tokens,
                    'count': generated_count
                })

        self.train()

        final_length = prompt.shape[1] + num_tokens
        return tokens[:, :final_length], layer_context

    def shift_memory(self, layers_out, stride):
        shifted = []
        for n, state in enumerate(layers_out):
            batch_size, total_len, hidden_dim = state.shape

            # 1. Grab the tokens being kept
            retained_state = state[:, stride:, :]

            # 2. Expand this layer's learned vector for the new empty slots
            pad = self.new_layer_context[n].expand(batch_size, stride, hidden_dim)

            # 3. Combine retained memory with the new padding
            new_state = torch.cat([retained_state, pad], dim=1)

            # 4. Detach the entire tensor so Window 1 starts completely clean
            shifted.append(new_state.detach())

        return shifted
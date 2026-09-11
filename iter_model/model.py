import torch
import torch.nn as nn

class FLB_Attention(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super(FLB_Attention, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)

        self.softmax = nn.Softmax(dim=-1)
        # self.out_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, fwd, lat, bck):
        batch_size, seq_len, _ = fwd.shape

        # Context merges all three streams
        context = fwd + lat + bck

        # Compute query, key, and value vectors
        Q = self.W_q(context)
        K = self.W_k(context)
        V = self.W_v(context)

        # Split the features into separate heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute raw dot product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Mask future tokens so earlier positions cannot cheat during generation
        mask = torch.triu(torch.ones(seq_len, seq_len, device=fwd.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        # Normalize scores and weight the values
        attn_weights = self.softmax(scores)
        context_out = torch.matmul(attn_weights, V)

        # Swap dimensions back and flatten heads into the original hidden dimension
        context_out = context_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        # context_out = self.out_proj(context_out)

        return context_out

class Layer_Block(nn.Module):
    def __init__(self, hidden_size, expansion=2, num_heads=4):
        super(Layer_Block, self).__init__()

        self.F = nn.Linear(hidden_size, hidden_size)
        self.L = nn.Linear(hidden_size, hidden_size)
        self.B = nn.Linear(hidden_size, hidden_size)

        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.B.bias)

        self.FLB_Attention = FLB_Attention(hidden_size, num_heads=num_heads)

        self.norm_fwd = nn.LayerNorm(hidden_size)
        self.norm_lat = nn.LayerNorm(hidden_size)
        self.norm_bck = nn.LayerNorm(hidden_size)

        self.norm_attn = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * expansion),
            nn.GELU(),
            nn.Linear(hidden_size * expansion, hidden_size)
        )

    def forward(self, fwd, lat, bck):
        # Let the model interpret incoming signals appropriately
        F = self.F(fwd)
        L = self.L(lat)
        B = self.B(bck)

        # Normalize incoming signals
        F = self.norm_fwd(F)
        L = self.norm_lat(L)
        B = self.norm_bck(B)

        # Process and combine all three directions through the activation
        attn_out = self.FLB_Attention(F, L, B)
        attn_out = self.norm_attn(attn_out)

        # Apply the feedforward network to the attention output
        update = self.ffn(L + attn_out)

        return update

class Iteration_Model(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_heads = 4, sweeps = 1, sweep_iters = 1, num_layers = 6, layer_iters = 1, window_size = 64):
        super(Iteration_Model, self).__init__()
        # model parameters
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.window_size = window_size

        # iteration parameters
        self.num_layers = num_layers
        self.layer_iters = layer_iters
        self.sweeps = sweeps
        self.sweep_iters = sweep_iters

        # A single learned canvas for future lookahead slots
        self.prediction_slot = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # For initializing new "empty" layers
        self.new_layer_context = nn.Parameter(
            torch.randn(num_layers, 1, 1, hidden_dim) * 0.02
        )
        
        # model layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(window_size * 2, hidden_dim) # Cannot predict more future tokens than are in the window
        self.layers = nn.ModuleList([Layer_Block(hidden_dim, expansion=2, num_heads=num_heads) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, layers_out=None, num_predictions=1):
        batch_size, seq_len = x.shape
        context_emb = self.embedding(x)

        # Attach future query slots if using multi-token prediction
        if num_predictions > 1:
            num_slots = num_predictions - 1
            query_emb = self.prediction_slot.expand(batch_size, num_slots, self.hidden_dim)
            fwd_stream = torch.cat([context_emb, query_emb], dim=1)
        else:
            fwd_stream = context_emb

        # Add positional embeddings across the entire n + p sequence
        total_len = fwd_stream.shape[1]
        positions = torch.arange(total_len, device=x.device)
        fwd_stream = fwd_stream + self.pos_embedding(positions)

        x_stream, layers_out = self.run_iterations(fwd_stream, layers_out, num_predictions)

        # Normalize and pass final output through prediction head
        x_stream = self.final_norm(x_stream)
        output = self.output(x_stream)
        return output, layers_out

    def run_iterations(self, x_stream, layers_out=None, num_predictions=1):
        batch_size, total_len, _ = x_stream.shape

        # If no previous memory exists, expand the learned initial states
        if layers_out is None:
            layers_out = [self.new_layer_context[n].expand(batch_size, total_len, self.hidden_dim) for n in range(self.num_layers)]

        for s in range(self.sweeps): # Go through each Sweep
            for si in range(self.sweep_iters): # Go through each Sweep Iteration
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
                        layers_out[n] = forward + self.layers[n](forward, layers_out[n], back)

                    ### end layer iterations
                ### end layers
            ### end sweep iterations

            # After every sweep update the input to the next sweep with the output of the last layer
            x_stream = layers_out[self.num_layers - 1]

        ### end sweeps

        return x_stream, layers_out

    def stream_model(self, prompt, layer_context, stride, num_predictions):
        # 1. Forward pass through the window
        output, layer_context = self.forward(prompt, layer_context, num_predictions)

        # 2. Extract predictions from the tail
        all_predictions = output[:, -num_predictions:, :]
        committed_outputs = all_predictions[:, :stride, :]
        speculative_predictions = all_predictions[:, stride:, :]

        # 3. Slide memory buffers left by stride
        next_layer_context = self.shift_memory(layer_context, stride)

        return committed_outputs, all_predictions, next_layer_context

    def train_model(self, sequence, optimizer=None, criterion=None, layer_context=None, stride = 1, num_predictions = 1, prediction_decay=True, min_pred_decay=0.25, accumulate_gradients=False, on_window=None):
        assert num_predictions >= stride, (
        f"Predictions ({num_predictions}) must be >= Stride ({stride})."
        )

        batch_size, seq_len = sequence.shape
        needed_len = self.window_size + num_predictions

        assert seq_len >= needed_len, (
            f"Sequence length ({seq_len}) is too short. "
            f"Must be at least {needed_len} tokens for window + predictions."
        )

        # Initialize memory if starting a new sequence
        if layer_context is None:
            canvas_len = self.window_size + max(0, num_predictions - 1)
            layer_context = [self.new_layer_context[n].expand(batch_size, canvas_len, self.hidden_dim) for n in range(self.num_layers)]

        # Calculate decay weights across all prediction slots
        weights = torch.ones(num_predictions, device=sequence.device)
        num_speculative = num_predictions - stride
    
        if prediction_decay and num_speculative > 0:
            for i in range(num_speculative):
                decay_factor = 1.0 - (1.0 - min_pred_decay) * (i / max(1, num_speculative - 1))
                weights[stride + i] = decay_factor
    
        slot_weights = weights.unsqueeze(0)
        weight_sum = weights.sum()
    
        if criterion is None:
            criterion = nn.CrossEntropyLoss(reduction='none')
    
        if optimizer is not None and not accumulate_gradients:
            optimizer.zero_grad()
    
        total_loss = 0.0
        window_count = 0

        # Slide across the sequence
        for start in range(0, seq_len - needed_len + 1, stride):
            input = sequence[:, start : start + self.window_size]
            targets = sequence[:, start + self.window_size : start + needed_len]

            # Step through 1 window of the model
            outputs, predictions, layer_context = self.stream_model(input, layer_context, stride, num_predictions)

            # Calculate token loss across all prediction slots
            raw_loss = criterion(
                predictions.reshape(-1, self.vocab_size),
                targets.reshape(-1)
            ).reshape(batch_size, -1)

            # Weigh loss based on decay rate
            step_loss = (raw_loss * slot_weights).sum() / (batch_size * weight_sum)

            # Backpropagation (doesnt update parameters yet)
            step_loss.backward()

            # Execute the diagnostic callback if provided
            if on_window is not None:
                on_window({
                    'window_idx': window_count,
                    'step_loss': step_loss.item(),
                    'input': input,
                    'outputs': outputs,
                    'predictions': predictions,
                    'targets': targets,
                    'raw_loss': raw_loss.detach()
                })

            # Update parameters if not accumulating them
            if not accumulate_gradients and optimizer is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += step_loss.item()
            window_count += 1

        # Final step update if accumulating across the sequence
        if accumulate_gradients and optimizer is not None and window_count > 0:
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.data.div_(window_count)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(1, window_count)
        return avg_loss, layer_context
    
    @torch.no_grad()
    def generate(self, prompt, num_tokens, temperature=0.8, stride=1, num_predictions=1, layer_context=None, on_step=None):
        self.eval()
        batch_size = prompt.shape[0]
        tokens = prompt.clone()
        generated_count = 0

        # Initialize recurrent memory if starting fresh
        if layer_context is None:
            layer_context = [self.new_layer_context[n].expand(batch_size, self.window_size + max(0, num_predictions - 1), self.hidden_dim) for n in range(self.num_layers)]

        while generated_count < num_tokens:
            if tokens.shape[1] < self.window_size:
                pad_len = self.window_size - tokens.shape[1]
                pad = torch.zeros(batch_size, pad_len, dtype=tokens.dtype, device=tokens.device)
                input = torch.cat([pad, tokens], dim=1)
            else:
                input = tokens[:, -self.window_size:]

            outputs, predictions, layer_context = self.stream_model(input, layer_context, stride, num_predictions)

            # Sample the output based on temperature
            probs = torch.softmax(outputs / temperature, dim=-1)
            sampled = torch.multinomial(probs.reshape(-1, self.vocab_size), num_samples=1).reshape(batch_size, stride)

            tokens = torch.cat([tokens, sampled], dim=1)
            generated_count += stride

            if on_step is not None:
                # Take the highest-probability character across every lookahead slot
                lookahead_tokens = torch.argmax(predictions, dim=-1)

                on_step({
                    'new_tokens': sampled,
                    'tokens_so_far': tokens,
                    'lookahead_tokens': lookahead_tokens,
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
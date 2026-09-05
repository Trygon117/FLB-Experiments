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

        self.FLB_Attention = FLB_Attention(hidden_size, num_heads=num_heads)

        self.norm_fwd = nn.LayerNorm(hidden_size)
        self.norm_lat = nn.LayerNorm(hidden_size)
        self.norm_bck = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * expansion),
            nn.GELU(),
            nn.Linear(hidden_size * expansion, hidden_size)
        )

    def forward(self, fwd, lat, bck):
        fwd_norm = self.norm_fwd(fwd)
        lat_norm = self.norm_lat(lat)
        bck_norm = self.norm_bck(bck)

        # Process and combine all three directions through the activation
        attn_out = self.FLB_Attention(self.F(fwd_norm), self.L(lat_norm), self.B(bck_norm))

        # Apply the feedforward network to the attention output
        update = self.ffn(attn_out)

        return lat + update

class Iteration_Model(nn.Module):
    def __init__(self, vocab_size, hidden_dim, max_seq_len=512, sweeps = 1, sweep_iters = 1, num_layers = 6, layer_iters = 1):
        super(Iteration_Model, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        self.layers = nn.ModuleList([Layer_Block(hidden_dim) for _ in range(num_layers)])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

        self.sweeps = sweeps
        self.sweep_iters = sweep_iters
        self.num_layers = num_layers
        self.layer_iters = layer_iters

    def forward(self, x):
        batch_size, seq_len = x.shape

        # get embeddings
        positions = torch.arange(seq_len, device=x.device)
        embeddings = self.embedding(x) + self.pos_embedding(positions)

        # A place to store the output of each layer for each sweep iteration
        layers_out = [torch.zeros_like(embeddings) for _ in range(self.num_layers)]

        x_stream = embeddings

        for s in range(self.sweeps): # Go through each Sweep
            for si in range(self.sweep_iters): # Go through each Sweep Iteration
                for n in range(self.num_layers): # Go through each layer

                    # Backward signal from the layer above
                    if n + 1 < self.num_layers:
                        back = layers_out[n+1]
                    else: 
                        back = torch.zeros_like(x_stream)

                    # Forward signal from the layer below or the base stream
                    if(n == 0):
                        forward = x_stream
                    else:
                        forward = forward + layers_out[n-1]

                    for l in range(self.layer_iters): # Go through every layer iteration

                        # Pass in the current forward input, the iterated lateral input, and the current backward input
                        layers_out[n] = self.layers[n](forward, layers_out[n], back)

                    ### end layer iterations
                ### end layers
            ### end sweep iterations

            # After every sweep update the input to the next sweep with the output of the last layer
            x_stream = x_stream + layers_out[self.num_layers - 1]
        ### end sweeps
        
        # Normalize and pass final output through prediction head
        x_stream = self.final_norm(x_stream)
        output = self.output(x_stream)
        return output

    
import torch 

class DummyGPTModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = torch.nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = torch.nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = torch.nn.Sequential(*[TransformerBlock(cfg) for _ in range (cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = torch.nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        s = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

class LayerNorm(torch.nn.Module):
    def __init__(self, emb_dim):
          super().__init__()
          self.eps = 1e-5
          self.scale = torch.nn.Parameter(torch.ones(emb_dim))
          self.shift = torch.nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepDim=True)
        var = x.var(dim=-1, keepDim=True, unbiased=False)
        norm_x = (x-mean)/ torch.sqrt(var + self.eps)                 
        return self.scale + norm_x + self.shift
    
    

class TransformerBlock(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__()
    
    def forward(self,x):
        return x
    

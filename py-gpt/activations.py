import torch 

class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) *(x + 0.044715 * torch.pow(x, 3))))
        


class FeedForward(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), GELU(),torch.nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
)

    def forward(self, x):
        return self.layers(x)
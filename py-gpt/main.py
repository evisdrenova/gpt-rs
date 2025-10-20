import torch
import tiktoken
from attention import MultiHeadAttention, SelfAttention
from model import DummyGPTModel
# from model import GPTModel, generate_text_simple,create_dataloader_v1, calc_loss_loader,train_model_simple

GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias
}


# with open('../the-verdict.txt',"r", encoding="utf-8") as f:
#     raw_text = f.read()


# tokenizer = tiktoken.get_encoding("gpt2")
# enc_text = tokenizer.encode(raw_text)

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)


torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)
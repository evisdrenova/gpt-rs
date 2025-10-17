import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import time
import re
# from model import GPTModel, generate_text_simple,create_dataloader_v1, calc_loss_loader,train_model_simple

with open('../the-verdict.txt',"r", encoding="utf-8") as f:
    raw_text = f.read()


tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(raw_text)

class GPTDatasetV1(Dataset):
    # stride is the number of positions the sliding window slides, since we're doign enxt token prediction, its just 1 usually
    # max_length is the max length of the input sequence
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        #creates the input and output tensors by iterating over tokenIDs 
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # returns the length of the data set
    def __len__(self):
        return len(self.input_ids)
    
    #returns a single row from the data set
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    
    return dataloader



input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 50257
output_dim = 256
max_length  = 4

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length,stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print("token embeddings shape:",token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("positional embedding shape:",pos_embeddings.shape)

# absolute positional encoding
input_embeddings = token_embeddings + pos_embeddings
print("input embeddings shape:",input_embeddings.shape)

# if __name__ == "__main__":
  
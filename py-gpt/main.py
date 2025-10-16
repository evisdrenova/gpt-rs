# import torch
# import tiktoken
import time
import re
# from model import GPTModel, generate_text_simple,create_dataloader_v1, calc_loss_loader,train_model_simple

with open('../the-verdict.txt',"r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

# if __name__ == "__main__":
  
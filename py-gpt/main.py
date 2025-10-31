import torch
import torch.nn as nn
import tiktoken
from attention import MultiHeadAttention, SelfAttention
from model import GPTModel, LayerNorm, TransformerBlock
from activations import GELU, FeedForward
from generation import generate, token_ids_to_text, text_to_token_ids
from dataset import (
    SpamDataSet,
    create_balanced_dataset,
    create_dataloader_v1,
    download_and_unzip_spam_data,
    random_split,
)
from loss import calc_loss_loader
from train import train_model_simple
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from load_weights import load_weights_into_gpt
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

# from gpt_download import download_and_load_gpt2

# settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

# print("Settings:", settings)
# print("Parameter dictionary keys:", params.keys())

# model_configs = {
#     "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
#     "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
#     "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
#     "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
# }

# model_name = "gpt2-small (124M)"
# NEW_CONFIG = GPT_CONFIG_124M.copy()
# NEW_CONFIG.update(model_configs[model_name])
# NEW_CONFIG.update({"qkv_bias": True})

# gpt = GPTModel(NEW_CONFIG)
# gpt.eval()

# load_weights_into_gpt(gpt, params)
# gpt.to("cpu")


tokenizer = tiktoken.get_encoding("gpt2")

# torch.manual_seed(123)
# token_ids = generate(
#     model=gpt,
#     idx=text_to_token_ids("Every effort moves you", tokenizer).to("cpu"),
#     max_new_tokens=25,
#     context_size=NEW_CONFIG["context_length"],
#     top_k=50,
#     temperature=1.5,
# )
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# zip_path = "sms_spam_collection.zip"
# extracted_path = "sms_spam_collection"
# data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"


# download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)


# df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
# print(df["Label"].value_counts())


# balanced_df = create_balanced_dataset(df)
# print(balanced_df["Label"].value_counts())
# balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

# train_df.to_csv("train.csv", index=None)
# validation_df.to_csv("validation.csv", index=None)
# test_df.to_csv("test.csv", index=None)

train_dataset = SpamDataSet(csv_file="train.csv", max_length=None, tokenizer=tokenizer)

print(train_dataset.max_length)

val_dataset = SpamDataSet(
    csv_file="validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer
)
test_dataset = SpamDataSet(
    csv_file="test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer
)

num_workers = 0
batch_size = 8
torch.manual_seed(123)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

for input_batch, target_batch in train_loader:
    pass
print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)

print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")

# file_path = "../the-verdict.txt"
# with open(file_path, "r", encoding="utf-8") as file:
#     text_data = file.read()

# tokenizer = tiktoken.get_encoding("gpt2")

# train_ratio = 0.90
# split_idx = int(train_ratio * len(text_data))
# train_data = text_data[:split_idx]
# val_data = text_data[split_idx:]


# train_loader = create_dataloader_v1(train_data,
# batch_size=2,
# max_length=GPT_CONFIG_124M["context_length"],
# stride=GPT_CONFIG_124M["context_length"],
# drop_last=True,
# shuffle=True,
# num_workers=0
# )
# val_loader = create_dataloader_v1(
# val_data,
# batch_size=2,
# max_length=GPT_CONFIG_124M["context_length"],
# stride=GPT_CONFIG_124M["context_length"],
# drop_last=False,
# shuffle=False,
# num_workers=0
# )


# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# # optimizer = torch.optim.AdamW(
# # model.parameters(),
# # lr=0.0004, weight_decay=0.1
# # )
# # num_epochs = 10
# # train_losses, val_losses, tokens_seen = train_model_simple(
# # model, train_loader, val_loader, optimizer, device,
# # num_epochs=num_epochs, eval_freq=5, eval_iter=5,
# # start_context="Every effort moves you", tokenizer=tokenizer
# # )

# token_ids = generate(
#     model=model,
#     idx=text_to_token_ids("Every effort moves you", tokenizer),
#     max_new_tokens=15,
#     context_size=GPT_CONFIG_124M["context_length"],
#     top_k=25,
#     temperature=1.4
# )
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


# # def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
# #     fig, ax1 = plt.subplots(figsize=(5, 3))
# #     ax1.plot(epochs_seen, train_losses, label="Training loss")
# #     ax1.plot(
# #     epochs_seen, val_losses, linestyle="-.", label="Validation loss"
# #     )
# #     ax1.set_xlabel("Epochs")
# #     ax1.set_ylabel("Loss")
# #     ax1.legend(loc="upper right")
# #     ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
# #     ax2 = ax1.twiny()
# #     ax2.plot(tokens_seen, train_losses, alpha=0)
# #     ax2.set_xlabel("Tokens seen")
# #     fig.tight_layout()
# #     plt.show()
# # epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# # plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

import collections
import math

import datasets
import numpy as np
import torch
import torch.nn as nn
import torchtext
import torchtext.data
import torchtext.vocab
import tqdm
from torch import optim
from torch.nn import TransformerEncoderLayer, TransformerEncoder

import wandb


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA device')

    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using MPS device')
    else:
        print('Using CPU')
        device = torch.device('cpu')
    return device


def tokenize_example(example, tokenizer, max_length):
    tokens = tokenizer(example["text"])[:max_length]
    length = len(tokens)
    return {"tokens": tokens, "length": length}


def create_vocab(data, min_freq=5):
    special_tokens = ["<unk>", "<pad>"]
    vocab = torchtext.vocab.build_vocab_from_iterator(
            data["tokens"],
            min_freq=min_freq,
            specials=special_tokens,
    )
    unk_index = vocab["<unk>"]
    pad_index = vocab["<pad>"]

    vocab.set_default_index(unk_index)
    return vocab, pad_index


def get_data_sets(max_length=256, test_size=0.25):
    train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

    train_data = train_data.map(
            tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
    )
    test_data = test_data.map(
            tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
    )

    train_valid_data = train_data.train_test_split(test_size=test_size)
    train_data = train_valid_data["train"]
    valid_data = train_valid_data["test"]

    return train_data, valid_data, test_data


def map_data_set(data, vocab):
    def numericalize_example(example, v):
        ids = v.lookup_indices(example["tokens"])
        return {"ids": ids}

    data = data.map(numericalize_example, fn_kwargs={"v": vocab})
    data = data.with_format(type="torch", columns=["ids", "label", "length"])

    return data


def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_length = [i["length"] for i in batch]
        batch_length = torch.stack(batch_length)
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "length": batch_length, "label": batch_label}
        return batch

    return collate_fn


def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader


def get_pretrained_embedding(vocab):
    vectors = torchtext.vocab.GloVe()
    return vectors.get_vecs_by_tokens(vocab.get_itos())


def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


def train(dataloader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(dataloader, desc="training..."):
        ids = batch["ids"].to(device)
        length = batch["length"]
        label = batch["label"].to(device)
        prediction = model(ids, length)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)


def evaluate(dataloader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            length = batch["length"]
            label = batch["label"].to(device)
            prediction = model(ids, length)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)


def train_epochs(nr_epochs, train_data_loader, valid_data_loader, model, criterion, optimizer, device):
    wandb.init(project="imdb-sentiment", config={'epochs': nr_epochs,
                                                 'model' : str(model)}
               )
    best_valid_loss = float("inf")
    for epoch in range(nr_epochs):
        train_loss, train_acc = train(
                train_data_loader, model, criterion, optimizer, device
        )
        valid_loss, valid_acc = evaluate(valid_data_loader, model, criterion, device)
        wandb.log({
            "train_loss": train_loss,
            "train_acc" : train_acc,
            "val_loss"  : valid_loss,
            "val_acc"   : valid_acc
        })

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "lstm.pt")
        print(f"epoch: {epoch}")
        print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
        print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")

    wandb.finish()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (max_len, d_model) to store positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate the positional encoding using sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register buffer so that it doesn't get updated during training
        self.pe = pe.unsqueeze(0)  # Shape becomes (1, max_len, d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # Add positional encoding to the input embedding
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x


class TransformerEncoderModule(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 embedding_dropout,
                 dropout,
                 pad_idx,
                 seq_length):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=12, dim_feedforward=hidden_dim, dropout=0.1,
                                             batch_first=True)
        self.encoder = TransformerEncoder(self.layer, num_layers=n_layers)

        # self.pooling = nn.AvgPool1d(seq_length)
        self.pooling = nn.AvgPool2d([seq_length, 1])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, output_dim)

        self.pos_encoding = PositionalEncoding(embedding_dim, seq_length)

    def forward(self, ids, length):
        embedding = self.embedding_dropout(self.pos_encoding(self.embedding(ids)))
        output = self.encoder.forward(embedding)
        output = self.pooling(output).squeeze(dim=1)
        output = self.fc(self.dropout(output))

        return output


def main():
    batch_size = 128
    max_length = 256
    test_size = 0.25
    min_freq = 5
    embedding_dim = 300
    hidden_dim = 300
    output_dim = 2
    n_layers = 3
    dropout_rate_emb = 0.0
    dropout_rate = 0.2
    nr_epochs = 30
    lr = 0.0001

    train_data, valid_data, test_data = get_data_sets(max_length, test_size)

    vocab, pad_index = create_vocab(train_data, min_freq)
    vocab_size = len(vocab)

    train_data = map_data_set(train_data, vocab)
    valid_data = map_data_set(valid_data, vocab)
    test_data = map_data_set(test_data, vocab)

    pretrained_embedding = get_pretrained_embedding(vocab)

    train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
    valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
    test_data_loader = get_data_loader(test_data, batch_size, pad_index)

    device = get_device()
    model = TransformerEncoderModule(vocab_size=vocab_size,
                                     embedding_dim=embedding_dim,
                                     hidden_dim=hidden_dim,
                                     output_dim=output_dim,
                                     n_layers=n_layers,
                                     embedding_dropout=dropout_rate_emb,
                                     dropout=dropout_rate,
                                     pad_idx=pad_index,
                                     seq_length=max_length)
    model.embedding.weight.data = pretrained_embedding
    model = model.to(device)


    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    train_epochs(nr_epochs, train_data_loader, valid_data_loader, model, criterion, optimizer, device)


if __name__ == "__main__":
    main()

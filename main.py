### main.py

import torch
from torch.utils.data import DataLoader, Dataset
from data_loader import load_conala
from vocab import get_tokenizers, build_vocab, SOS, EOS
from model import Seq2Seq
from train import train_epoch
from evaluate import predict
import os

class CodeDataset(Dataset):
    def __init__(self, data, vocab, src_tok, tgt_tok):
        self.data = data
        self.vocab = vocab
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_tokens = [SOS] + self.src_tok(src) + [EOS]
        tgt_tokens = [SOS] + self.tgt_tok(tgt) + [EOS]
        src_tensor = torch.tensor([self.vocab[token] for token in src_tokens])
        tgt_tensor = torch.tensor([self.vocab[token] for token in tgt_tokens])
        return src_tensor, tgt_tensor

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    srcs = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    tgts = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=0)
    return srcs, tgts

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_conala()
    src_tok, tgt_tok = get_tokenizers()
    vocab = build_vocab(data, src_tok, tgt_tok)

    dataset = CodeDataset(data, vocab, src_tok, tgt_tok)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = Seq2Seq(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    if os.path.exists("best_model.pth"):
        print("🔁 Loading pre-trained model...")
        model.load_state_dict(torch.load("best_model.pth"))
        model.eval()
    else:
        print("🔨 Training model...")
        best_loss = float("inf")
        for epoch in range(30):
            loss = train_epoch(model, loader, optimizer, criterion, device)
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), "best_model.pth")
                print("✅ Saved better model")
        model.eval()

    # Interactive loop
    while True:
        prompt = input("\nEnter natural language (or 'exit'): ")
        if prompt.lower() == "exit":
            break
        code = predict(model, prompt, vocab, src_tok, device)
        print(f"\nGenerated Code:\n{code}")

if __name__ == "__main__":
    main()

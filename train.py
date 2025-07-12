import torch
from torch.amp import autocast, GradScaler
import torch.nn as nn
import torch.optim as optim
from model import TransformerModel
from tokenizer import encode
from pathlib import Path
import datetime
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VOCAB_SIZE = 32000
MODEL_PATH = Path("checkpoints/transformer.pt")
LOG_PATH = Path("logs/train_log.json")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

model = TransformerModel(vocab_size=VOCAB_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

src_text = "Hello world!"
tgt_text = "Bonjour le monde !"

src_ids = torch.tensor([encode(src_text)], dtype=torch.long).to(device)
tgt_ids = torch.tensor([encode(tgt_text)], dtype=torch.long).to(device)
tgt_input = tgt_ids[:, :-1]
tgt_target = tgt_ids[:, 1:]

model.train()
optimizer.zero_grad()

with autocast(device_type='cuda', dtype=torch.float16):
    logits = model(src_ids, tgt_input)
    loss = criterion(logits.view(-1, logits.size(-1)), tgt_target.view(-1))

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

torch.save(model.state_dict(), MODEL_PATH)
print(f"Модель сохранена в {MODEL_PATH}")

log_data = {
    "datetime": datetime.datetime.now().isoformat(),
    "src": src_text,
    "tgt": tgt_text,
    "loss": float(loss.item())
}
with open(LOG_PATH, "w") as f:
    json.dump(log_data, f, indent=4)

print(f"Лог сохранён в {LOG_PATH}")
print(f"Loss: {loss.item():.4f}")
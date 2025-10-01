from transformers import AutoTokenizer, AutoModelForCausalLM, PerceiverModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PerceiverModel, PerceiverConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

# Use Qwen tokenizer for tokens
model_name = "Qwen/Qwen3-4B-Instruct-2507"
perpection_model = PerceiverModel.from_pretrained("deepmind/multimodal-perceiver", torch_dtype="auto", device_map="auto")
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
qwen = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", load_in_4bit=True)
emb_layer = qwen.get_input_embeddings()
vocab_size = tok.vocab_size

class PerceiverPreprocessor(nn.Module):
    """
    Standalone preprocessor that converts embeddings (e.g. 2560-dim tokens)
    into the 768-dim inputs required by a Perceiver.
    """

    def __init__(self, input_dim=2560, output_dim=768, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        x: [batch, seq_len, input_dim] or [batch, input_dim]
        returns: [batch, seq_len, output_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # add seq_len=1 if missing
        return self.proj(x)

class PerceiverMind(nn.Module):
    def __init__(self,
                 pretrained=True,
                 mind_latent_dim=2560,
                 mind_num_latents=1024,
                 n_heads=8,
                 n_layers=1):
        super().__init__()

        # Perceiver backbone (sensory encoder)
        if pretrained:
            self.perceiver = perpection_model
        else:
            config = PerceiverConfig()
            self.perceiver = PerceiverModel(config)

        d_latents = self.perceiver.config.d_latents

        # Persistent "mind" state (trainable initialization)
        self.mind_latents = nn.Parameter(
            torch.randn(mind_num_latents, mind_latent_dim)
        )

        # Projectors to align sensory and mind dims
        self.sensory_proj = nn.Linear(d_latents, mind_latent_dim)

        # Cross-attention block(s): mind queries sensory features
        self.cross_attn = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=mind_latent_dim,
                nhead=n_heads,
                dim_feedforward=4*mind_latent_dim,
                batch_first=True
            ) for _ in range(n_layers)
        ])

        # Readout head
        self.head = nn.Linear(mind_latent_dim, 128)  # adjust to task

    def forward(self, x, mind_state=None):
        """
        x: dict/tensor for Perceiver input
        mind_state: [batch, num_latents, dim] persistent state
        """

        # Encode input with Perceiver
        perceiver_out = self.perceiver(**x)
        sensory_latents = perceiver_out.last_hidden_state  # [batch, n_latents, d_latents]

        # Project to mind-dim
        sensory_latents = self.sensory_proj(sensory_latents)  # [batch, n_latents, mind_latent_dim]

        # Init mind state if none
        if mind_state is None:
            B = sensory_latents.size(0)
            mind_state = self.mind_latents.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention update: mind attends to sensory
        # (treat sensory as "context memory")
        for layer in self.cross_attn:
            # concat mind + sensory, but only update mind tokens
            combined = torch.cat([mind_state, sensory_latents], dim=1)
            updated = layer(combined)
            mind_state = updated[:, :mind_state.size(1), :]  # take back just the mind part

        # Normalize mind state to avoid drift
        mind_state = nn.functional.layer_norm(mind_state, mind_state.shape[-1:])

        # Readout (e.g., classification, regression, etc.)
        output = self.head(mind_state.mean(dim=1))

        return output, mind_state

class PerceiverDecoder(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, latents, labels=None):
        pooled = latents.mean(dim=1)  # [B, d_model]
        logits = self.fc(self.ln(pooled))  # [B, vocab_size]

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return logits, loss


D_MODEL = 2560
pre = PerceiverPreprocessor(input_dim=D_MODEL, output_dim=768).to(device)
model = PerceiverMind(pretrained=False, mind_num_latents=1024).to(device)
decoder = PerceiverDecoder(d_model=D_MODEL, vocab_size=vocab_size).to(device)

from torch.utils.data import DataLoader
from datasets import load_dataset

dataset = load_dataset("imdb")

def tokenize_batch(batch):
    return tok(batch["text"])
dataset = dataset.map(tokenize_batch, batched=True)

train_data = dataset["train"]
test_data = dataset["test"]

# torch DataLoader
def collate_fn(batch):
    ids = torch.tensor([item["input_ids"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return {"input_ids": ids, "labels": labels}

train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collate_fn)

import torch.nn.functional as F

def event_reconstruction_loss(latents, event_embeds):
    with torch.no_grad():
        # latents: [B, N, D], event_embeds: [B, E, D]
        pooled = latents.mean(dim=1)  # [B, D]
        target = event_embeds.mean(dim=1)  # average event embedding
        return F.mse_loss(pooled, target)

def next_event_prediction_loss(latents, next_event_embeds, temperature=0.1):
    with torch.no_grad():
        # latents: [B, N, D], next_event_embeds: [B, D]
        pooled = latents.mean(dim=1)  # [B, D]
        pooled = F.normalize(pooled, dim=-1)
        targets = F.normalize(next_event_embeds, dim=-1)

        logits = pooled @ targets.T.squeeze(1) / temperature   # [B, B]
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

def slot_diversity_loss(latents):
    with torch.no_grad():
        # latents: [B, N, D]
        B, N, D = latents.shape
        latents = F.normalize(latents, dim=-1)  # cosine sim
        sim = latents @ latents.transpose(-1, -2)  # [B, N, N]
        eye = torch.eye(N, device=latents.device).unsqueeze(0)
        return ((sim - eye) ** 2).mean()

def sparsity_loss(attn_weights):
    with torch.no_grad():
        # attn_weights: [B, N] (after softmax for each event)
        return attn_weights.mean(dim=0).sum()  # encourage few slots active

def stability_loss(latents, prev_latents):
    with torch.no_grad():
        return F.mse_loss(latents, prev_latents)

optimizer = torch.optim.AdamW(
    list(pre.parameters()) + list(model.parameters()), lr=1e-4
)

from torch.amp import autocast

# Training step
for batch in train_loader:  # batch["input_ids"], batch["labels"]
    ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
    with torch.no_grad():
        tok_embeds = emb_layer(ids)
    loss_total = 0
    mind_state = None
    with autocast(dtype=torch.bfloat16, device_type=device):
        proj = pre(tok_embeds)
    for t in range(ids.size(1) - 1):
        with autocast(dtype=torch.bfloat16, device_type=device):
            # Cast token embeddings to bfloat16 to match latents dtype
            token = tok_embeds[:, t:t+1, :]  # current token embedding
            next_token = tok_embeds[:, t+1:t+2, :]if t < ids.size(1) - 2 else None
            target = ids[:, t+1]                                      # next token ID
            proj_vector = proj[:, t:t+1]
            x_t = {"inputs": proj_vector, "attention_mask": None}
            prev_mind_state = mind_state if mind_state is not None else None
            out, mind_state = model(x_t, mind_state)
            mind_state = mind_state.detach().clone()

            # Decode prediction
            _, loss = decoder(mind_state, labels=target)

            # Main loss
            lm_loss = loss  # your decoder head loss

            # Auxiliary
            rec_loss  = event_reconstruction_loss(mind_state, token)
            pred_loss = next_event_prediction_loss(mind_state, next_token) if next_token is not None else 0
            div_loss  = slot_diversity_loss(mind_state)
            stab_loss = stability_loss(mind_state, prev_mind_state) if prev_mind_state is not None else 0

            # Weighted sum
            loss = lm_loss \
                + 0.1 * rec_loss \
                + 0.1 * pred_loss \
                + 0.01 * div_loss \
                + 0.001 * stab_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_total += loss.item()

    print(f"Average loss per token: {loss_total / (ids.size(1)-1):.4f}")
import torch
import torch.nn as nn
from transformers import PerceiverModel, PerceiverConfig, PreTrainedModel, PretrainedConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
import os

class CrossAttentionLayer(nn.Module):
    """Proper cross-attention: queries from one source, keys/values from another"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, queries, context):
        """
        queries: [batch, num_queries, d_model] - the mind state
        context: [batch, num_context, d_model] - the sensory input
        """
        # Cross attention
        attn_out, _ = self.multihead_attn(
            query=queries,
            key=context,
            value=context
        )
        queries = self.norm1(queries + attn_out)
        
        # FFN
        ffn_out = self.ffn(queries)
        queries = self.norm2(queries + ffn_out)
        
        return queries

class PerceiverMindConfig(PretrainedConfig):
    model_type = "perceiver_mind"

    def __init__(self,
                 mind_latent_dim=2560,
                 mind_num_latents=1024,
                 n_heads=8,
                 n_layers=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.mind_latent_dim = mind_latent_dim
        self.mind_num_latents = mind_num_latents
        self.n_heads = n_heads
        self.n_layers = n_layers

class PerceiverMind(PreTrainedModel):
    config_class = PerceiverMindConfig
    
    def __init__(self, config):
        super().__init__(config)

        # Perceiver backbone (sensory encoder)

        self.perceiver = PerceiverModel.from_pretrained("deepmind/multimodal-perceiver", torch_dtype="auto")

        d_latents = self.perceiver.config.d_latents

        # Better initialization for mind latents (small values)
        self.mind_latents = nn.Parameter(
            torch.randn(config.mind_num_latents, config.mind_latent_dim) * 0.02
        )

        # Project sensory to mind dimension
        self.sensory_proj = nn.Sequential(
            nn.Linear(d_latents, config.mind_latent_dim),
            nn.LayerNorm(config.mind_latent_dim)
        )

        # Proper cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=config.mind_latent_dim,
                n_heads=config.n_heads,
                dropout=0.1
            ) for _ in range(config.n_layers)
        ])

        # Gating mechanism for state updates (like samskara formation)
        self.update_gate = nn.Linear(config.mind_latent_dim * 2, config.mind_latent_dim)

        # Readout head
        self.head = nn.Linear(config.mind_latent_dim, 128)  # adjust to task

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

        # Clone state to allow residual connection
        prev_state = mind_state

        # Update mind state via cross-attention
        # Mind queries the sensory input to "perceive"
        for layer in self.cross_attn_layers:
            mind_state = layer(queries=mind_state, context=sensory_latents)

        # Learnable gating: how much to update vs retain
        # (like skandhas being impressed upon citta)
        gate = torch.sigmoid(self.update_gate(torch.cat([prev_state, mind_state], dim=-1)))
        mind_state = gate * mind_state + (1 - gate) * prev_state

        # Readout (e.g., classification, regression, etc.)
        output = self.head(mind_state.mean(dim=1))

        return output, mind_state


device = "cuda" if torch.cuda.is_available() else "cpu"

perception_model = PerceiverMind.from_pretrained("Bossologist/perception-model-test", device_map="auto", torch_dtype="auto")

model_name = "Qwen/Qwen3-4B-Instruct-2507"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
qwen = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True
)
emb_layer = qwen.get_input_embeddings()

special_tokens_dict = {"additional_special_tokens": ["<null>"]}
num_added = tok.add_special_tokens(special_tokens_dict)

if num_added > 0:
    qwen.resize_token_embeddings(len(tok))

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 2) Prepare for k-bit training (enables input grads, fixes LN cast, etc.)
qwen = prepare_model_for_kbit_training(qwen, use_gradient_checkpointing=True)
qwen.gradient_checkpointing_enable()  # memory saver

# 3) LoRA config (targets match LLaMA/Qwen3 module names)
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Common Qwen3 module names (adjust if your printout shows slightly different):
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",   # attention
        "up_proj","down_proj","gate_proj"      # MLP
    ],
)

from peft import get_peft_model

qwen = get_peft_model(qwen, lora_cfg)
qwen.print_trainable_parameters()

encoder_layer = nn.Linear(2560, 704).to(qwen.device)
conversion_layer = nn.Linear(768, 2560).to(qwen.device)

params = list(perception_model.parameters()) + list(qwen.parameters())
opt = torch.optim.AdamW(params, lr=1e-5)

def dropout_schedule(batch_proportion):
    return min(1.0, batch_proportion + 0.1)

import torch.nn.functional as F

def get_loss(res_embeds, target):
    return F.mse_loss(res_embeds, target)

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DataCollatorWithPadding

collator = DataCollatorWithPadding(tokenizer=tok, return_tensors="pt")

dataset = load_dataset("Bossologist/general_Qwen3_ft_dataset")

def tokenize_batch(batch):
    # Remove "<think>\n\n</think>\n" from each text before tokenization
    cleaned_texts = [text.replace("<think>\n\n</think>\n", "") for text in batch["text"]]
    return tok(cleaned_texts)
dataset = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])

train_data = dataset["train"].shuffle(seed=42).select(range(5000))
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collator)

with torch.no_grad():
    null_embed = qwen.get_input_embeddings()(torch.tensor([tok.convert_tokens_to_ids("<null>")]).to(qwen.device))[0]

def train_step(batch, latents, batch_proportion: float):
    """ 
    texts: list of strings (B)
    latents: torch.Tensor of shape (B, N, 2560)
    """

    # Encode input with Perceiver
    ids = batch["input_ids"].to(qwen.device)                  # [B, T]
    with torch.no_grad():
        tok_embeds = emb_layer(ids).detach() # [B, T, H]

    # (Optional) remove positions for updater: we pass raw token embeddings (no pos enc)
    # Update citta with full sequence at once (you can also stream token by token)
    with torch.amp.autocast(dtype=torch.bfloat16, device_type=device):
        # Token dropout
        p = dropout_schedule(batch_proportion)
        drop_mask = (torch.rand_like(ids.float()) < p).unsqueeze(-1)  # [B,T,1]
        tok_embeds_masked = tok_embeds.masked_fill(drop_mask, 0.0)

        silent = True
        curr_turn = "system"
        for t in range(ids.size(1) - 1):
            curr_token = tok_embeds[:, t:t+1, :]       # current token

            curr_id = ids[:, t]
            if curr_id == tok.convert_tokens_to_ids("<|im_end|>"):
                if curr_turn == "system":
                    curr_turn = "user"
                elif curr_turn == "user":
                    curr_turn = "assistant"
                    silent = False
                else:
                    curr_turn = "system"
                    silent = True

            _, latents = perception_model({"inputs": encoder_layer(curr_token), "attention_mask": None}, latents)     # update state

            target = tok_embeds[:, t+1] if not silent else null_embed.expand(tok_embeds.size(0), -1)  # next token embedding
            latents_proj = conversion_layer(latents)

            # Forward with latents + token
            inputs_embeds = torch.cat([latents_proj, tok_embeds_masked[:, :t, :]], dim=1)
            outputs = qwen(inputs_embeds=inputs_embeds, output_hidden_states=True)
            res_embeds = outputs.hidden_states[-1][:, -1, :]  # [b, 2560]
            loss = get_loss(res_embeds, target)
            loss.backward()
            latents = latents.detach()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    print("Average loss: ", loss.item()/ids.size(1))

    return float(loss.item())


for batch_idx, batch in enumerate(train_loader):
    train_step(batch, None, batch_idx / len(train_loader))
    print(f"Batch {batch_idx} completed out of {len(train_loader)}")

qwen.push_to_hub("Bossologist/Qwen3-4B-Instruct-2507_enden_ft", token=os.getenv("HF_KEY"))
tok.push_to_hub("Bossologist/Qwen3-4B-Instruct-2507_enden_ft", token=os.getenv("HF_KEY"))
perception_model.push_to_hub("Bossologist/perception-model-test_enden_ft", token=os.getenv("HF_KEY"))

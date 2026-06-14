# src/memory/builder_3b.py (fixed dtype issue)
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Tuple
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class QuantizedBuilder3B(nn.Module):
    """
    Builder module f_θ: Updates implicit belief state b_t from observations.
    Uses LoRA-adapted Qwen2.5-3B for efficient encoding.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.belief_dim = config.get("belief_dim", 768)
        self.hidden_dim = config.get("hidden_dim", 256)

        # Initialize base model with 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config["quantization"]["load_in_4bit"],
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=config["quantization"]["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=config["quantization"]["bnb_4bit_use_double_quant"],
        )

        print("Loading base model for Builder...")
        self.base_model = AutoModel.from_pretrained(
            config["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        # Apply LoRA
        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            target_modules=config["lora"]["target_modules"],
            lora_dropout=config["lora"]["dropout"],
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )

        print("Applying LoRA to Builder...")
        self.model = get_peft_model(self.base_model, lora_config)

        # Freeze base model parameters, only LoRA is trainable
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["base_model"],
            trust_remote_code=True,
            padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get the dtype from the base model
        self.model_dtype = next(self.model.parameters()).dtype

        # Gated update mechanism (GRU-style) - use same dtype
        self.gate_update = nn.GRUCell(self.belief_dim, self.belief_dim)

        # Projection layer for session embeddings - match model dtype
        self.session_proj = nn.Linear(self.base_model.config.hidden_size, self.belief_dim)

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(self.belief_dim)

        # Move projection layers to same device and dtype as model
        self.session_proj = self.session_proj.to(dtype=self.model_dtype)
        self.gate_update = self.gate_update.to(dtype=self.model_dtype)
        self.layer_norm = self.layer_norm.to(dtype=self.model_dtype)

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Builder initialized. Trainable parameters: {trainable}")

    def encode_session(self, session_text: str) -> torch.Tensor:
        """Encode a single session into an embedding."""
        inputs = self.tokenizer(
            session_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config["training"]["max_seq_length"],
            padding=True
        )

        # Move inputs to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling over sequence dimension
            session_embedding = outputs.last_hidden_state.mean(dim=1)

        # Project to belief dimension, ensuring same dtype
        session_embedding = session_embedding.to(dtype=self.model_dtype)
        projected = self.session_proj(session_embedding)

        return projected

    def forward(self, belief_state: torch.Tensor, session_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update belief state with new observation.

        Args:
            belief_state: Previous belief state b_{t-1} [batch_size, belief_dim]
            session_text: New observation o_t

        Returns:
            Updated belief state b_t, Session embedding
        """
        session_embedding = self.encode_session(session_text)

        # Ensure belief_state has correct dtype
        if belief_state.dtype != self.model_dtype:
            belief_state = belief_state.to(dtype=self.model_dtype)

        # Gated update: b_t = GRU(b_{t-1}, session_embedding)
        updated_belief = self.gate_update(session_embedding, belief_state)
        updated_belief = self.layer_norm(updated_belief)

        return updated_belief, session_embedding

    def get_trainable_parameters(self):
        """Return only trainable parameters for optimizer."""
        # Return LoRA parameters and the projection layers
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        proj_params = [self.session_proj.weight, self.session_proj.bias] if self.session_proj.bias is not None else [
            self.session_proj.weight]
        gate_params = list(self.gate_update.parameters())
        norm_params = list(self.layer_norm.parameters())

        return lora_params + proj_params + gate_params + norm_params

    def train(self, mode=True):
        """Set training mode."""
        super().train(mode)
        self.model.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        super().eval()
        self.model.eval()
        return self

    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.model.to(device)
        self.gate_update.to(device)
        self.session_proj.to(device)
        self.layer_norm.to(device)
        return self
from __future__ import annotations

from typing import Dict

import torch

from .backbone import BackboneBase
from src.models.modeling_graphformers import GraphFormers
from src.models.tnlrv3.configuration_tnlrv3 import TuringNLRv3Config


class GraphformersBackbone(BackboneBase):
    """GraphFormers backbone for AMR graph encoding."""

    def __init__(self, config, hidden_size: int = 768):
        super().__init__()
        self.model = GraphFormers(config)  # ← Changé ici
        self.output_dim = hidden_size

    @classmethod
    def from_pretrained(cls, ckpt_path: str = None) -> GraphformersBackbone:
        config = TuringNLRv3Config(
            vocab_size=28996,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            output_hidden_states=True,
            rel_pos_bins=0,
            max_rel_pos=128,
        )
        instance = cls(config, hidden_size=config.hidden_size)

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            instance.model.load_state_dict(state_dict, strict=False)
        else:
            from transformers import BertModel
            bert_state_dict = BertModel.from_pretrained("bert-base-uncased").state_dict()

            # Filtrer les clés dont la shape ne correspond pas
            model_state_dict = instance.model.state_dict()  # ← Plus besoin de .bert
            filtered_state_dict = {
                k: v for k, v in bert_state_dict.items()
                if k in model_state_dict and v.shape == model_state_dict[k].shape
            }

            instance.model.load_state_dict(filtered_state_dict, strict=False)

            loaded = set(filtered_state_dict.keys())
            total = set(model_state_dict.keys())
            print(f"Loaded {len(loaded)}/{len(total)} parameter tensors from bert-base-uncased")

        return instance

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        B, N, L = x["input_ids"].shape
        D = self.output_dim

        # Reshape pour GraphFormers : (B, N, L) → (B*N, L)
        input_ids = x["input_ids"].view(B * N, L)
        attention_mask = x["attention_mask"].view(B * N, L)
        neighbor_mask = x["neighbor_mask"]  # (B, N)

        # Forward pass
        encoder_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            neighbor_mask=neighbor_mask,
        )

        # Extraire l'embedding CLS (position 1 après le station token)
        hidden_states = encoder_outputs[0]  # (B*N, L+1, D)
        cls_embeddings = hidden_states[:, 1, :].view(B, N, D)  # (B, N, D)

        # Retourner l'embedding du nœud principal (index 0)
        node_embeddings = cls_embeddings[:, 0, :]  # (B, D)

        return node_embeddings
import random
from typing import Dict, List, Tuple

import penman
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
import json
import logging
logging.getLogger("penman").setLevel(logging.WARNING)

class AMRGraphDataset(Dataset):
    """
    Dataset of AMR nodes. Each sample is a list of N view dicts for one center
    node, where views[0..num_global_views-1] are high-coverage global views
    (teacher + student) and views[num_global_views..] are low-coverage local
    views (student only).

    This directly mirrors DINO multi-crop: global = full-res, local = small crop.

    Args:
        amr_file:            Path to file of Penman AMR graphs (one per line or block).
        tokenizer_name:      HuggingFace tokenizer identifier.
        neighbor_num:        Max neighbors to include per view (padding applied if fewer).
        token_length:        Max token length per node text sequence.
        num_global_views:    Number of high-coverage views (teacher + student).
        num_local_views:     Number of low-coverage views (student only).
        global_neighbor_rate: Fraction of neighbors to keep for global views.
        local_neighbor_rate:  Fraction of neighbors to keep for local views.
    """

    def __init__(
        self,
        amr_file: str,
        tokenizer_name: str = "bert-base-uncased",
        neighbor_num: int = 10,
        token_length: int = 30,
        num_global_views: int = 2,
        num_local_views: int = 4,
        local_neighbor_rate: float = 0.4,
        bfs_depth: int = 1,
    ):
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.neighbor_num = neighbor_num
        self.token_length = token_length
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views
        self.local_neighbor_rate = local_neighbor_rate
        self.bfs_depth = bfs_depth

        # Parse all AMR graphs and build (center_node, neighbors_list) records
        self.records: List[Tuple[str, List[str]]] = []
        self._parse(amr_file)

    def _bfs_neighbors(
        self,
        graph: penman.Graph,
        start: str,
        node_concept: Dict[str, str],
    ) -> List[str]:
        visited = {start}
        queue: List[Tuple[str, int]] = [(start, 0)]
        result: List[str] = []

        while queue:
            current_id, depth = queue.pop(0)

            if depth >= self.bfs_depth:
                continue

            for src, role, target in graph.triples:
                if role == ":instance":
                    continue
                if src == current_id:
                    tgt_label = node_concept.get(target, str(target))
                    result.append(f"{role} {tgt_label}")
                    if target in node_concept and target not in visited:
                        visited.add(target)
                        queue.append((target, depth + 1))

        return result

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self, amr_file: str) -> None:
        """Read AMR file and build flat list of (concept, [edge+neighbor]) records."""
        with open(amr_file) as json_data:
            data = json.load(json_data)
            print(f"Loaded {len(data)} AMR graphs from {amr_file}")

        graphs: List[penman.Graph] = []
        cpt_error = []
        for i, graph in enumerate(data):
            try:
                decoded_graph = penman.decode(graph)
                graphs.append(decoded_graph)
            except Exception as e:
                cpt_error.append(i)
                continue
        
        print(f"Decoded {len(graphs)} AMR graphs, skipped {len(cpt_error)} due to errors: {cpt_error}")

        for graph in graphs:
            node_concept = {inst.source: inst.target for inst in graph.instances()}
            
            for node_id, concept in node_concept.items():
                nbrs = self._bfs_neighbors(graph, node_id, node_concept)
                if nbrs:
                    self.records.append((concept, nbrs))

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def _encode_node_texts(self, texts: List[str]) -> Tuple[List[List[int]], List[List[int]]]:
        """Tokenize a list of node text strings. Returns (input_ids, attention_masks)."""
        encoded = self.tokenizer(
            texts,
            max_length=self.token_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def _build_view(
        self, center_concept: str, neighbor_texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Build a single graph view dict.

        Returns:
            {
                "input_ids":      (N_total, L) padded to neighbor_num+1 rows
                "attention_mask": (N_total, L)
                "neighbor_mask":  (N_total,)   — 1 for real nodes, 0 for padding
            }
        """
        # Center node is always first; neighbors follow
        all_texts = [center_concept] + neighbor_texts

        # Truncate to max neighbors (center + neighbor_num)
        all_texts = all_texts[: self.neighbor_num + 1]

        input_ids, attention_mask = self._encode_node_texts(all_texts)

        real_n = len(all_texts)
        total_n = self.neighbor_num + 1  # center + max neighbors

        # Pad with zeros to fixed size
        pad_rows = total_n - real_n
        pad_seq = [[0] * self.token_length] * pad_rows
        input_ids = input_ids + pad_seq
        attention_mask = attention_mask + pad_seq

        neighbor_mask = [1.0] * real_n + [0.0] * pad_rows

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),            # (N, L)
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),  # (N, L)
            "neighbor_mask": torch.tensor(neighbor_mask, dtype=torch.float),   # (N,)
        }

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> List[Dict[str, torch.Tensor]]:
        center_concept, all_neighbors = self.records[idx]
        views = []
        n = len(all_neighbors)

        # Global views: All neighbors with random shuffle (Perte info profondeur, mais evite redondance) peut etre 
        for _ in range(self.num_global_views):
            nbrs = random.sample(all_neighbors, n)
            views.append(self._build_view(center_concept, nbrs))

        # Local views : 40% neighbors randomly choosed
        for _ in range(self.num_local_views):
            k = max(1, int(n * self.local_neighbor_rate))
            nbrs = random.sample(all_neighbors, min(k, len(all_neighbors)))
            views.append(self._build_view(center_concept, nbrs))

        return views  # List[Dict], length = num_global_views + num_local_views
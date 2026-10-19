
from pathlib import Path

import numpy as np
from torch_geometric.datasets import Planetoid

BASE_DIR = Path(__file__).resolve().parent
dataset = Planetoid(root=str(BASE_DIR / "data" / "Planetoid"), name="Cora")
data = dataset[0]

output_path = BASE_DIR / "cora_data.npz"

np.savez(
    output_path,
    x=data.x.cpu().numpy().astype(np.float32),
    edge_index=data.edge_index.cpu().numpy().astype(np.int64),
)

print(f"Saved dataset to: {output_path}")
print(f"Node features: {data.x.shape}")
print(f"Edge index: {data.edge_index.shape}")

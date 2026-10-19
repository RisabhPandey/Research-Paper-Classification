
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "simple_gcn_cora.onnx"
DATA_PATH = BASE_DIR / "cora_data.npz"
STATIC_DIR = BASE_DIR / "static"

FEATURE_DIMENSION = 1433
NUM_CLASSES = 7

CORA_CLASSES = {
    0: "Case-Based",
    1: "Genetic Algorithms",
    2: "Neural Networks",
    3: "Probabilistic Methods",
    4: "Reinforcement Learning",
    5: "Rule Learning",
    6: "Theory",
}


# --------------------------------------------------
# Load ONNX model with conservative CPU settings
# --------------------------------------------------

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")

session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
session_options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
)

model_session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=session_options,
    providers=["CPUExecutionProvider"],
)


# --------------------------------------------------
# Load preprocessed Cora arrays once
# No PyTorch / PyTorch Geometric at runtime
# --------------------------------------------------

if not DATA_PATH.is_file():
    raise FileNotFoundError(
        "cora_data.npz is missing. Create it using the "
        "preprocessing script provided with this deployment."
    )

with np.load(DATA_PATH, allow_pickle=False) as data:
    CORA_FEATURES = np.asarray(data["x"], dtype=np.float32)
    CORA_EDGE_INDEX = np.asarray(data["edge_index"], dtype=np.int64)

if CORA_FEATURES.ndim != 2 or CORA_FEATURES.shape[1] != FEATURE_DIMENSION:
    raise ValueError("Cora features must have shape (num_nodes, 1433).")

if CORA_EDGE_INDEX.ndim != 2 or CORA_EDGE_INDEX.shape[0] != 2:
    raise ValueError("Cora edge_index must have shape (2, num_edges).")

if (
    CORA_EDGE_INDEX.size
    and (
        CORA_EDGE_INDEX.min() < 0
        or CORA_EDGE_INDEX.max() >= CORA_FEATURES.shape[0]
    )
):
    raise ValueError("Cora edge_index contains an invalid node index.")


# --------------------------------------------------
# FastAPI application and request schemas
# --------------------------------------------------

app = FastAPI(title="Research Paper Classification API")


class GraphPredictRequest(BaseModel):
    node_features: List[List[float]]
    edge_index: Optional[List[List[int]]] = None


class CoraNodeRequest(BaseModel):
    node_indices: List[int] = Field(min_length=1, max_length=50)


# --------------------------------------------------
# Inference helpers
# --------------------------------------------------

def softmax(scores: np.ndarray) -> np.ndarray:
    """Convert logits into numerically stable probabilities."""
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def run_model(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    node_indices_to_return: List[int],
) -> dict:
    """Run ONNX inference and return predictions for selected nodes."""

    node_features = np.ascontiguousarray(node_features, dtype=np.float32)
    edge_index = np.ascontiguousarray(edge_index, dtype=np.int64)

    try:
        logits = model_session.run(
            ["logits"],
            {
                "node_features": node_features,
                "edge_indices": edge_index,
            },
        )[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(exc)}",
        ) from exc

    probabilities = softmax(logits)
    predicted_classes = np.argmax(logits, axis=-1)

    predictions = []
    for index in node_indices_to_return:
        class_id = int(predicted_classes[index])
        predictions.append(
            {
                "node_index": index,
                "predicted_class_id": class_id,
                "predicted_class_name": CORA_CLASSES[class_id],
                "probabilities": probabilities[index].tolist(),
                "logits": logits[index].tolist(),
            }
        )

    return {
        "num_nodes": int(node_features.shape[0]),
        "num_edges": int(edge_index.shape[1]),
        "prediction": predictions,
    }


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def home_page():
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))

    return {"service": "Simple GCN Cora API", "status": "healthy"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "providers": model_session.get_providers(),
    }


@app.get("/model-info")
def model_info():
    return {
        "Model Name": "SimpleGCN",
        "feature_dimension": FEATURE_DIMENSION,
        "num_classes": NUM_CLASSES,
        "class_mapping": CORA_CLASSES,
        "input": [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in model_session.get_inputs()
        ],
        "outputs": [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in model_session.get_outputs()
        ],
        "cora_nodes": int(CORA_FEATURES.shape[0]),
        "cora_edges": int(CORA_EDGE_INDEX.shape[1]),
    }


@app.post("/predict")
def predict_custom_graph(request: GraphPredictRequest):
    if not request.node_features:
        raise HTTPException(
            status_code=400,
            detail="Node features cannot be empty.",
        )

    node_features = np.asarray(request.node_features, dtype=np.float32)

    if node_features.ndim != 2 or node_features.shape[1] != FEATURE_DIMENSION:
        raise HTTPException(
            status_code=422,
            detail=f"Each node feature vector must have {FEATURE_DIMENSION} dimensions.",
        )

    if not np.isfinite(node_features).all():
        raise HTTPException(
            status_code=422,
            detail="Node features must contain only finite numbers.",
        )

    num_nodes = node_features.shape[0]

    if request.edge_index is not None:
        edge_index = np.asarray(request.edge_index, dtype=np.int64)

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise HTTPException(
                status_code=422,
                detail="edge_index must have shape (2, num_edges).",
            )

        if edge_index.size and (
            edge_index.min() < 0 or edge_index.max() >= num_nodes
        ):
            raise HTTPException(
                status_code=422,
                detail="edge_index contains a node index outside the graph.",
            )
    else:
        node_indices = np.arange(num_nodes, dtype=np.int64)
        edge_index = np.vstack((node_indices, node_indices))

    return run_model(
        node_features,
        edge_index,
        list(range(num_nodes)),
    )


@app.post("/predict/cora_node")
def predict_real_cora_node(request: CoraNodeRequest):
    invalid_indices = [
        index
        for index in request.node_indices
        if index < 0 or index >= CORA_FEATURES.shape[0]
    ]

    if invalid_indices:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Node indices must be between 0 and "
                f"{CORA_FEATURES.shape[0] - 1}. "
                f"Invalid indices: {invalid_indices}"
            ),
        )

    return run_model(
        CORA_FEATURES,
        CORA_EDGE_INDEX,
        request.node_indices,
    )


# Mount static assets last so API routes remain available.
if STATIC_DIR.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(STATIC_DIR), html=True),
        name="static",
    )

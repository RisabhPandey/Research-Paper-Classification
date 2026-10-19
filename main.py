"""
We are going to create our API now.

"""

import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from torch_geometric.datasets import Planetoid
from fastapi.staticfiles import StaticFiles



"""
The model can predict 7 possible topics for each paper.
Their is a dictionary that checks which label belongs to which topic.

"""

CORA_CLASSES = {
    0: "Case-Based",
    1: "Genetic Algorithms",
    2: "Neural Networks",
    3: "Probabilistic Methods",
    4: "Reinforcement Learning",
    5: "Rule Learning",
    6: "Theory"
}

" Figure out where this file lives, so we can find the model file "

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "simple_gcn_cora.onnx")
DATA_DIR = os.path.join(BASE_DIR, "data","Planetoid")
STATIC_DIR = os.path.join(BASE_DIR, "static")

"""
Load the trained model once, when the app starts up
"""
model_session = ort.InferenceSession(MODEL_PATH, providers = ['CPUExecutionProvider'])


app = FastAPI()

"""
Request Formats : These describes what JSON data user must send us

"""
class GraphPredictRequest(BaseModel):
    """
    This is the request format for the graph prediction endpoint.
    It contains the graph data in a specific format.
    """

    # The node features should be a list of lists of floats
    node_features: List[List[float]]  

    # Optional: The edge index should be a list of lists of integers
    edge_index: Optional[List[List[int]]] = None


class CoraNodeRequest(BaseModel):
    """
    This is the request format for the Cora node prediction endpoint.
    It contains the node features and edge index in a specific format.
    """

    # The node (by index number) from the real cora dataset that we want to predict the topic for
    node_indices: List[int]


"""
Helper Function
"""
def softmax(scores: np.ndarray):
    # Turns the raw model score(logits) into a probability that comes b/w 0 and 1
    # we will get the max probability 

    shifted = scores.max(axis =1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)


def run_model(node_features: np.ndarray, edge_index: np.ndarray, node_indices_to_return):
    """
    Run ONX model only on the graph we are given & return its prediction
    """
    output = model_session.run(
        ["logits"],
        {
            'node_features':node_features.astype(np.float32),
            'edge_indices':edge_index.astype(np.int64)
        },
    )
    logits = output[0]
    probabilities = softmax(logits)
    predicted_classes = logits.argmax(axis = -1)

    result = []

    for i in node_indices_to_return:
        result.append({
            "node_index"            :i,
            "predicted_class_id"    :int(predicted_classes[i]),
            "predicted_class_name"  :CORA_CLASSES[int(predicted_classes[i])],
            "probabilities"         :probabilities[i].tolist(),
            "logits"                :logits[i].tolist(),
        }
        )

    return {
        'num_nodes': node_features.shape[0],
        'num_edges':edge_index.shape[1],
        'prediction':result
    }



# UI connecting  
@app.get("/")
def home_page():
    """
    Our UI will be shown
    """
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"Service":"Simple GCN Cora API","status":"healthy"}



# Creating Routes
@app.get("/health")
def health_check():
    return {"status":"healthy","providers":model_session.get_providers()}

@app.get("/model-info")
def model_info():
    """
            Shows basic details about model - I/P, O/P, classes
      """
    return{
        "Model Name":"SimpleGCN",
        "feature_dimension":1433,
        "num_classes":7,
        "class_mapping":CORA_CLASSES,
        "input":[
            {"name":inp.name,"shape":inp.shape,"type":inp.type}
            for inp in model_session.get_inputs()
        ],
        "outputs":[
            {"name":out.name,"shape":out.shape,"type":out.type}
            for out in model_session.get_outputs()
        ],
    }


@app.post("/predict")
def predict_custom_graph(request: GraphPredictRequest):
    """
    Classify a graph you provide yourself & optionally your own edges/connections
    """
    if request.node_features == 0:
        raise HTTPException(status_code=400, detail="Node features cannot be empty.")

    for feature_vector in request.node_features:
        if len(feature_vector) != 1433:
            raise HTTPException(status_code=422, detail="Each node feature vector must have 1433 dimensions.")


    node_features = np.array(request.node_features, dtype=np.float32)
    num_nodes = len(node_features)

    if request.edge_index:            # When we are getting indices from the user
        edge_index = np.array(request.edge_index, dtype=np.int64)
        if edge_index.shape[0] != 2 or edge_index.ndim != 2:
            raise HTTPException(status_code=422, detail="Edge index must have shape (2, num_edges).")

    else:                           # When we are not getting indices from the user.
        # no edge given: just connect every node to itself (self-loop)
        node_indices = np.arange(num_nodes, dtype=np.int64)
        edge_index = np.vstack([node_indices, node_indices])

    all_node_features = list(range(num_nodes))
    return run_model(node_features, edge_index,all_node_features)
        

@app.post("/predict/cora_node")
def predict_real_cora_node(request: CoraNodeRequest):
    """
    Classify a node from the real cora dataset
    """
    from torch_geometric.datasets import Planetoid
    try:
        cora_dataset = Planetoid(root=DATA_DIR, name='Cora')[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading Cora dataset: {str(e)}")

    largest_valid_index = cora_dataset.num_nodes - 1
    invalid_index =[
        i for i in request.node_indices
        if i < 0 or i > largest_valid_index
    ]
    if invalid_index:
        raise HTTPException(
            400, f"node_indices out of bound (must be 0 to {largest_valid_index}). Invalid indices: {invalid_index}"
        )

    return run_model(
        cora_dataset.x.numpy(),
        cora_dataset.edge_index.numpy(),
        request.node_indices
    )


if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR , html = True), name="static")

    
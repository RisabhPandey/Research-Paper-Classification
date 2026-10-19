# Research Paper Classification

An interactive web application that classifies nodes from the **Cora
citation network** into seven research-paper topic categories. The
project uses a Graph Convolutional Network (GCN) exported to **ONNX**, a
**FastAPI** backend, and a web dashboard for submitting node IDs and
viewing predictions.

## Live Demo

-   **Web app:**
    https://research-paper-classification-xtec.onrender.com/
-   **GitHub repository:**
    https://github.com/RisabhPandey/Research-Paper-Classification
-   **API documentation:**
    https://research-paper-classification-xtec.onrender.com/docs
-   **Health endpoint:**
    https://research-paper-classification-xtec.onrender.com/health
-   **Model information:**
    https://research-paper-classification-xtec.onrender.com/model-info

## Features

-   Predict a topic for one or more Cora node IDs.
-   Display predicted classes and model confidence scores.
-   Explore the seven Cora topic categories in the dashboard.
-   Try predictions for five random papers.
-   Filter results by topic.
-   Export prediction results as CSV.
-   Inspect the API through FastAPI's interactive Swagger documentation.
-   Run model inference with ONNX Runtime on CPU.
-   Use a preprocessed NumPy dataset file to avoid loading PyTorch
    Geometric during API inference.

## Topic Categories

The model maps its seven output classes to these labels:

    Class ID Topic
  ---------- ------------------------
           0 Case-Based
           1 Genetic Algorithms
           2 Neural Networks
           3 Probabilistic Methods
           4 Reinforcement Learning
           5 Rule Learning
           6 Theory

## Technology Stack

-   **Python**
-   **FastAPI** --- API routes and backend
-   **ONNX Runtime** --- model inference
-   **NumPy** --- dataset arrays and probability calculations
-   **Pydantic** --- request validation
-   **HTML, CSS, JavaScript** --- dashboard
-   **PyTorch Geometric** --- used locally to prepare the Cora dataset
-   **Render** --- deployment

## How It Works

1.  A user enters one or more Cora node IDs in the dashboard.
2.  The frontend sends the IDs to the FastAPI endpoint
    `POST /predict/cora_node`.
3.  The backend selects the preprocessed Cora node features and graph
    edges.
4.  ONNX Runtime runs the GCN model on CPU.
5.  The API returns predicted class IDs, topic names, logits, and
    probabilities.
6.  The dashboard displays the prediction results.

``` text
User
 |
 v
Web Dashboard (HTML / CSS / JavaScript)
 |
 v
FastAPI API
 |
 v
Preprocessed Cora graph (cora_data.npz)
 |
 v
ONNX Runtime + GCN model (simple_gcn_cora.onnx)
 |
 v
Predicted topic + probabilities
 |
 v
Dashboard results
```

## Project Structure

``` text
Research-Paper-Classification/
├── main.py                  # FastAPI application and prediction endpoints
├── prepare_cora.py          # Locally prepares the Cora arrays
├── cora_data.npz            # Preprocessed node features and graph edges
├── simple_gcn_cora.onnx     # Exported GCN model
├── requirements.txt         # Runtime dependencies
├── .python-version          # Python version used for deployment
└── static/
    ├── index.html           # Dashboard markup
    ├── styles.css           # Dashboard styling
    └── script.js            # Dashboard interactions and API calls
```

Other local folders, such as notebooks or downloaded dataset files, may
also be present in the repository.

## Run Locally

### 1. Clone the repository

``` bash
git clone https://github.com/RisabhPandey/Research-Paper-Classification.git
cd Research-Paper-Classification
```

### 2. Create and activate a virtual environment

Windows PowerShell:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

``` bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install the packages listed in `requirements.txt`:

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Ensure the data and model files exist

The API expects these files in the repository root:

-   `simple_gcn_cora.onnx`
-   `cora_data.npz`

If `cora_data.npz` is missing, prepare it locally in an environment
where PyTorch Geometric is installed. Install the additional preparation
dependency if necessary:

``` bash
pip install torch torch-geometric
```

Then run:

``` bash
python prepare_cora.py
```

The script saves the Cora node features and edge index to
`cora_data.npz`. This is a one-time preparation step; the deployed API
uses the saved arrays and does not need to load Cora through PyTorch
Geometric for each prediction.

### 5. Start the API

``` bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Open the local dashboard at http://127.0.0.1:8000/ and API documentation
at http://127.0.0.1:8000/docs.

## API Endpoints

  -----------------------------------------------------------------------
  Method                  Endpoint                Purpose
  ----------------------- ----------------------- -----------------------
  `GET`                   `/`                     Serves the dashboard

  `GET`                   `/health`               Reports service health
                                                  and ONNX providers

  `GET`                   `/model-info`           Returns model
                                                  input/output metadata
                                                  and class mapping

  `POST`                  `/predict/cora_node`    Predicts topics for
                                                  selected Cora node IDs

  `POST`                  `/predict`              Runs inference on a
                                                  supplied graph and its
                                                  node features
  -----------------------------------------------------------------------

### Example: Predict Cora nodes

Request:

``` http
POST /predict/cora_node
Content-Type: application/json
```

``` json
{
  "node_indices": [142, 1024, 2707]
}
```

The node IDs must be within the Cora dataset's valid range. The API
returns a prediction entry for each requested node, including its
predicted class ID, class name, probabilities, and logits.

### Example: Health check

``` http
GET /health
```

The response includes a service status and the ONNX Runtime execution
providers.

### Interactive API documentation

Visit `/docs` on the running application to inspect endpoints and submit
test requests without writing a separate client.

## Deployment

This project is deployed on Render:

https://research-paper-classification-xtec.onrender.com/

The service is configured to use ONNX Runtime with the CPU execution
provider. The preprocessed dataset reduces runtime overhead by avoiding
PyTorch Geometric dataset loading inside prediction requests. Free
hosting plans have resource limits, so memory usage and deployment logs
should still be monitored.

## Notes and Limitations

-   This model classifies nodes from the **Cora citation-network
    dataset**; it does not directly read an arbitrary research-paper PDF
    or infer a topic from raw paper text.
-   The model's confidence score is a model output, not a guarantee that
    a prediction is correct.
-   The custom `/predict` endpoint expects node feature vectors with
    1,433 features each. Meaningful graph-based predictions depend on
    providing graph data appropriate for the model.
-   The app uses CPU inference, which is suitable for a small demo but
    may not be ideal for high-volume production workloads.

## Future Improvements

-   Add automated tests for API validation and inference.
-   Add model evaluation metrics and a documented evaluation procedure.
-   Improve error reporting and request-size limits.
-   Add reproducible model-training and ONNX-export instructions.
-   Add screenshots or a short demo GIF to the repository README.

## Author

**Risabh Pandey**

-   GitHub: https://github.com/RisabhPandey

------------------------------------------------------------------------

If you find this project useful, consider giving the repository a ⭐ on
GitHub.

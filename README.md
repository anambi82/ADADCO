# ADADCO Network Intrusion Detection System

**CSCE 482 Capstone Project**

A full-stack network intrusion detection system that uses machine learning to analyze network traffic and detect various types of attacks.

## Quick Start Guide

Follow these steps in order to get the system running:

### 1. Run Docker Environment

First, set up the Jupyter notebook environment using Docker:

```bash
# Build the Docker image
docker build -t jupyter-notebook .

# Run the container (Mac/Linux)
docker run -p 8888:8888 -v $(pwd):/app jupyter-notebook

# Run the container (Windows Command Prompt)
docker run -p 8888:8888 -v %cd%:/app jupyter-notebook

# Run the container (Windows PowerShell)
docker run -p 8888:8888 -v ${PWD}:/app jupyter-notebook
```

Open your browser and navigate to `http://localhost:8888` to access Jupyter.

### 2. Train the Model

In the Jupyter environment, run the model training notebook:

1. Open `modelV2.ipynb`
2. Run all cells to train the ML model
3. This will generate the required artifacts:
   - `artifacts/mlp.pt` - Trained ML model
   - `artifacts/scaler_stats.npz` - Normalization statistics  
   - `artifacts/classes.txt` - Class labels

### 3. Run Frontend and Backend

#### Terminal 1 - Backend Server
```bash
cd /Users/ansonthai/School/Capstone/ADADCO
pip install -r requirements.txt
python backend/app.py
```

Expected output:
```
INFO:__main__:Loaded 13 classes
INFO:__main__:Loaded scaler with 10 features
INFO:__main__:Loaded model: 10 input features -> 13 classes
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8000
```

#### Terminal 2 - Frontend Server
```bash
cd /Users/ansonthai/School/Capstone/ADADCO/frontend
npm install
npm start
```

Expected output:
```
Compiled successfully!
You can now view frontend in the browser.
  Local:            http://localhost:3000
```

### 4. Use the Application

1. **Access the Web Interface**: Open `http://localhost:3000` in your browser

2. **Upload and Analyze Files**:
   - Go to the "File Upload" tab
   - Click "Choose a file" and select a CSV file from your `datasets/` directory
   - Click "Upload" to analyze the file
   - Wait for analysis to complete (30-60 seconds for large files)

3. **View Results**:
   - Switch to the "Data Visualization" tab
   - View analysis summary with anomaly counts and rates
   - See attack type distribution charts
   - Review detailed analysis results

## System Architecture

- **Backend**: Flask API server with PyTorch ML model (port 8000)
- **Frontend**: React web application for file upload and visualization (port 3000)
- **ML Model**: Trained neural network from modelV2.ipynb

## Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- Docker
- At least 4GB RAM (for ML model processing)

## Expected CSV File Format

Your CSV files should have these 10 features (in any order):
- Flow Packets/s, Flow Duration, Flow IAT Mean, Idle Mean, Fwd Packets/s
- Flow IAT Max, Total Fwd Packets, Packet Length Variance, Init_Win_bytes_forward, SYN Flag Count

**Plus a Label column** with values like: BENIGN, DDoS, PortScan, Bot, DoS Hulk, etc.

## Troubleshooting

### Backend Won't Start
```bash
# Check if artifacts exist
ls artifacts/
# If missing, run modelV2.ipynb to generate them

# Check Python dependencies
pip install -r requirements.txt
```

### Frontend Won't Start
```bash
# Clear npm cache and reinstall
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### Model Loading Errors
```bash
# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"
```

For detailed troubleshooting and API documentation, see `RUN_INSTRUCTIONS.md`.

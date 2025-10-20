# Flask Backend API Documentation

Network intrusion detection ML model serving API built with Flask.

## Quick Start

```bash
# Install dependencies
pip install -r ../requirements.txt

# Run the server
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### 1. Health Check
**GET** `/health`

Check if the server and model are loaded properly.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-10-15T10:30:00.000000"
}
```

---

### 2. Model Information
**GET** `/model/info`

Get information about the loaded model and classes.

**Response:**
```json
{
  "classes": ["BENIGN", "Bot", "DDoS", "DoS Hulk", ...],
  "num_classes": 12,
  "input_features": 78,
  "device": "cpu",
  "model_architecture": {
    "type": "MLP",
    "hidden_size": 156
  }
}
```

---

### 3. Single/Multi Prediction
**POST** `/predict`

Make predictions on one or more samples.

**Request Body:**
```json
{
  "data": [
    [0.5, 1.2, -0.3, ...],  // Feature array (78 features)
    [0.8, 0.9, 0.1, ...]
  ],
  "return_probabilities": false  // Optional: return class probabilities
}
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": "BENIGN",
      "prediction_index": 0,
      "confidence": 0.987,
      "probabilities": {  // Only if return_probabilities=true
        "BENIGN": 0.987,
        "Bot": 0.005,
        "DDoS": 0.003,
        ...
      }
    }
  ],
  "count": 1
}
```

**Frontend Example:**
```javascript
const response = await fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    data: [[0.5, 1.2, -0.3, /* ...78 features */]],
    return_probabilities: true
  })
});
const result = await response.json();
console.log(result.predictions[0].prediction); // "BENIGN"
```

---

### 4. Model Evaluation
**POST** `/evaluate`

Evaluate model performance with labeled data. Returns comprehensive metrics.

**Request Body:**
```json
{
  "data": [
    [0.5, 1.2, -0.3, ...],
    [0.8, 0.9, 0.1, ...]
  ],
  "labels": [0, 2]  // Class indices (0=BENIGN, 1=Bot, 2=DDoS, etc.)
}
```

**Response:**
```json
{
  "overall_metrics": {
    "accuracy": 0.95,
    "precision": 0.94,
    "recall": 0.95,
    "f1_score": 0.945
  },
  "per_class_metrics": [
    {
      "class": "BENIGN",
      "precision": 0.98,
      "recall": 0.97,
      "f1_score": 0.975,
      "support": 1000
    },
    ...
  ],
  "confusion_matrix": [
    [980, 10, 5, ...],
    [15, 920, 8, ...],
    ...
  ],
  "sample_count": 1000
}
```

**Frontend Example:**
```javascript
const response = await fetch('http://localhost:5000/evaluate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    data: testFeatures,
    labels: testLabels
  })
});
const metrics = await response.json();
// Display metrics.overall_metrics.accuracy, etc.
```

---

### 5. File Upload for Anomaly Detection ⭐
**POST** `/analyze_file`

**This is the main endpoint for your frontend file upload feature!**

Upload a CSV file containing network traffic data and get detailed anomaly analysis. Perfect for detecting attacks in network logs.

**Request:**
- **Method:** POST with `multipart/form-data`
- **File field name:** `file`
- **Supported format:** CSV files with 78 feature columns

**Response:**
```json
{
  "success": true,
  "filename": "network_traffic.csv",
  "analysis": {
    "total_samples": 10000,
    "benign_count": 8500,
    "anomaly_count": 1500,
    "anomaly_rate": 15.0
  },
  "attack_breakdown": {
    "DDoS": {
      "count": 800,
      "percentage": 8.0
    },
    "Bot": {
      "count": 400,
      "percentage": 4.0
    },
    "PortScan": {
      "count": 300,
      "percentage": 3.0
    }
  },
  "attack_confidence": {
    "DDoS": 0.95,
    "Bot": 0.89,
    "PortScan": 0.92
  },
  "top_anomalies": [
    {
      "sample_index": 523,
      "attack_type": "DDoS",
      "confidence": 0.98,
      "probabilities": {
        "BENIGN": 0.01,
        "DDoS": 0.98,
        "Bot": 0.01
      }
    }
  ],
  "summary": {
    "most_common_attack": "DDoS",
    "unique_attack_types": 3,
    "severity": "MEDIUM"
  }
}
```

**Frontend Example (React):**
```javascript
function FileUploadComponent() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/analyze_file', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input type="file" accept=".csv" onChange={handleFileUpload} />
      {loading && <p>Analyzing file...</p>}
      {results && (
        <div>
          <h3>Analysis Results for {results.filename}</h3>
          <p>Total Samples: {results.analysis.total_samples}</p>
          <p>Anomalies Detected: {results.analysis.anomaly_count} ({results.analysis.anomaly_rate}%)</p>
          <p>Severity: {results.summary.severity}</p>

          <h4>Attack Breakdown:</h4>
          <ul>
            {Object.entries(results.attack_breakdown).map(([attack, data]) => (
              <li key={attack}>
                {attack}: {data.count} ({data.percentage.toFixed(2)}%)
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:5000/analyze_file \
  -F "file=@network_traffic.csv"
```

**CSV Format Requirements:**
- 78 columns (features)
- No header row (or will be treated as data)
- Numeric values only
- Each row represents one network connection/packet

---

### 6. Batch Prediction
**POST** `/batch_predict`

Process large batches of data efficiently. Accepts CSV files or JSON arrays.

**Option A - JSON Request:**
```json
{
  "data": [
    [0.5, 1.2, -0.3, ...],
    [0.8, 0.9, 0.1, ...],
    // ...thousands of samples
  ]
}
```

**Option B - CSV Upload:**
```javascript
const formData = new FormData();
formData.append('file', csvFile);

fetch('http://localhost:5000/batch_predict', {
  method: 'POST',
  body: formData
});
```

**Response:**
```json
{
  "predictions": ["BENIGN", "DDoS", "Bot", ...],
  "prediction_counts": {
    "BENIGN": 850,
    "DDoS": 100,
    "Bot": 50
  },
  "total_samples": 1000
}
```

---

## Error Responses

All endpoints return appropriate HTTP status codes and error messages:

```json
{
  "error": "Expected 78 features, got 50"
}
```

Common status codes:
- `200` - Success
- `400` - Bad Request (invalid input)
- `500` - Internal Server Error (model not loaded, processing error)

---

## Data Format Notes

1. **Feature Count**: The model expects exactly **78 features** per sample
2. **Scaling**: Features are automatically scaled using the loaded scaler
3. **Class Labels**: Use class indices (0-11) for evaluation endpoint
4. **Batch Size**: Batch predictions are processed in chunks of 1000 for memory efficiency

---

## Example Frontend Integration

### React Example
```javascript
import { useState } from 'react';

function PredictionComponent() {
  const [result, setResult] = useState(null);

  const predictData = async (features) => {
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        data: [features],
        return_probabilities: true
      })
    });
    const data = await response.json();
    setResult(data.predictions[0]);
  };

  return (
    <div>
      {result && (
        <>
          <p>Prediction: {result.prediction}</p>
          <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
        </>
      )}
    </div>
  );
}
```

### Metrics Dashboard Example
```javascript
const fetchMetrics = async (testData, testLabels) => {
  const response = await fetch('http://localhost:5000/evaluate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      data: testData,
      labels: testLabels
    })
  });
  return await response.json();
};

// Use metrics for charts
const metrics = await fetchMetrics(data, labels);
displayAccuracyChart(metrics.overall_metrics.accuracy);
displayConfusionMatrix(metrics.confusion_matrix);
displayPerClassMetrics(metrics.per_class_metrics);
```

---

## Testing the API

### Using curl

```bash
# Health check
curl http://localhost:5000/health

# Get model info
curl http://localhost:5000/model/info

# Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[0.5, 1.2, -0.3, ...]],
    "return_probabilities": true
  }'
```

### Using Python
```python
import requests

# Make prediction
response = requests.post('http://localhost:5000/predict', json={
    'data': [[0.5, 1.2, -0.3, ...]],  # 78 features
    'return_probabilities': True
})
result = response.json()
print(f"Prediction: {result['predictions'][0]['prediction']}")
print(f"Confidence: {result['predictions'][0]['confidence']}")
```

---

## Class Labels Reference

| Index | Class Name |
|-------|-----------|
| 0 | BENIGN |
| 1 | Bot |
| 2 | DDoS |
| 3 | DoS Hulk |
| 4 | DoS Slowhttptest |
| 5 | DoS slowloris |
| 6 | FTP-Patator |
| 7 | Infiltration |
| 8 | PortScan |
| 9 | Web Attack – Brute Force |
| 10 | Web Attack – Sql Injection |
| 11 | Web Attack – XSS |

---

## Troubleshooting

**Model not loading:**
- Ensure `artifacts/mlp.pt`, `artifacts/scaler_stats.npz`, and `artifacts/classes.txt` exist
- Check the console logs for specific error messages

**Feature count mismatch:**
- Verify your input data has exactly 78 features
- Check the scaler was trained with the same feature set

**CORS issues:**
- CORS is enabled for all origins by default
- Adjust CORS settings in `app.py` if needed

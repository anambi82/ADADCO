from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import io
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Model definition (matches your notebook)
class MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        hidden = max(32, min(512, in_dim * 2))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Global variables for model and artifacts
model = None
scaler_stats = None
classes = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global storage for latest analysis results (for visualization endpoints)
latest_analysis = {
    'results': None,
    'timestamp': None,
    'filename': None
}

def load_model_artifacts():
    """Load model, scaler stats, and classes from artifacts directory"""
    global model, scaler_stats, classes

    try:
        # Get the artifacts directory
        artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')

        # Load classes
        classes_path = os.path.join(artifacts_dir, 'classes.txt')
        with open(classes_path, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(classes)} classes")

        # Load scaler statistics (manual normalization from your notebook)
        scaler_path = os.path.join(artifacts_dir, 'scaler_stats.npz')
        scaler_data = np.load(scaler_path)
        scaler_stats = {
            'mean': scaler_data['mean'],
            'std': scaler_data['std'],
            'columns': scaler_data['columns']
        }
        logger.info(f"Loaded scaler with {len(scaler_stats['mean'])} features")

        # Load model
        model_path = os.path.join(artifacts_dir, 'mlp.pt')
        checkpoint = torch.load(model_path, map_location=device)

        # Initialize model with correct dimensions
        in_dim = len(scaler_stats['mean'])
        num_classes = len(classes)
        model = MLP(in_dim, num_classes)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        logger.info(f"Loaded model: {in_dim} input features -> {num_classes} classes")

        return True
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        return False

def manual_normalize(data):
    """Manual normalization matching your notebook's approach"""
    global scaler_stats
    
    if scaler_stats is None:
        raise ValueError("Scaler stats not loaded")
    
    # Convert to DataFrame for easier handling
    if isinstance(data, np.ndarray):
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame(data)
    
    # Fill missing values
    df = df.fillna(0.0)
    
    # Apply Z-score normalization: (x - mean) / std
    mean = scaler_stats['mean']
    std = scaler_stats['std']
    
    # Ensure std is not zero (prevent division by zero)
    std = np.where(std == 0, 1, std)
    
    normalized = (df.values - mean) / std
    
    return normalized.astype(np.float32)

def calculate_accuracy(y_true, y_pred):
    """Manual accuracy calculation"""
    return np.mean(y_true == y_pred)

def calculate_precision_recall_f1(y_true, y_pred, num_classes):
    """Manual precision, recall, F1 calculation"""
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    support_per_class = []
    
    for class_idx in range(num_classes):
        # True positives, false positives, false negatives
        tp = np.sum((y_true == class_idx) & (y_pred == class_idx))
        fp = np.sum((y_true != class_idx) & (y_pred == class_idx))
        fn = np.sum((y_true == class_idx) & (y_pred != class_idx))
        support = np.sum(y_true == class_idx)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
        support_per_class.append(support)
    
    # Weighted averages
    total_support = np.sum(support_per_class)
    weights = np.array(support_per_class) / total_support if total_support > 0 else np.ones(num_classes) / num_classes
    
    weighted_precision = np.average(precision_per_class, weights=weights)
    weighted_recall = np.average(recall_per_class, weights=weights)
    weighted_f1 = np.average(f1_per_class, weights=weights)
    
    return {
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1': weighted_f1,
        'per_class': {
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1': f1_per_class,
            'support': support_per_class
        }
    }

def confusion_matrix_manual(y_true, y_pred, num_classes):
    """Manual confusion matrix calculation"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_class, pred_class in zip(y_true, y_pred):
        cm[true_class][pred_class] += 1
    return cm

# Load artifacts on startup
load_model_artifacts()

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - API information"""
    return jsonify({
        'name': 'Network Intrusion Detection API',
        'version': '1.0.0',
        'status': 'running',
        'model_loaded': model is not None,
        'endpoints': {
            'GET /': 'API information (this page)',
            'GET /health': 'Health check',
            'GET /model/info': 'Get model information',
            'POST /predict': 'Make predictions on data',
            'POST /evaluate': 'Evaluate model with labeled data',
            'POST /batch_predict': 'Batch predictions (CSV or JSON)',
            'POST /analyze_file': 'Upload CSV file for anomaly detection analysis',
            'GET /analysis/summary': 'Get summary of latest analysis',
            'GET /analysis/attacks': 'Get attack type distribution',
            'GET /analysis/anomalies': 'Get detailed anomaly information (supports query params)',
            'GET /analysis/confidence': 'Get confidence metrics by attack type',
            'GET /analysis/timeline': 'Get time-series anomaly data (supports query params)'
        },
        'documentation': 'See README.md for detailed API documentation',
        'github': 'https://github.com/anambi82/ADADCO'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler_stats is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    return jsonify({
        'classes': classes,
        'num_classes': len(classes),
        'input_features': len(scaler_stats['mean']) if scaler_stats else 0,
        'device': str(device),
        'model_architecture': {
            'type': 'MLP',
            'hidden_size': model.net[0].out_features if model else 0
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions on input data"""
    if model is None or scaler_stats is None:
        return jsonify({'error': 'Model or scaler not loaded'}), 500

    try:
        data = request.get_json()

        if 'data' not in data:
            return jsonify({'error': 'Missing "data" field in request'}), 400

        input_data = np.array(data['data'])
        return_probs = data.get('return_probabilities', False)

        # Validate input shape
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        expected_features = len(scaler_stats['mean'])
        if input_data.shape[1] != expected_features:
            return jsonify({
                'error': f'Expected {expected_features} features, got {input_data.shape[1]}'
            }), 400

        # Manual normalization
        normalized_data = manual_normalize(input_data)

        # Convert to tensor
        X_tensor = torch.FloatTensor(normalized_data).to(device)

        # Predict
        with torch.no_grad():
            logits = model(X_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        # Convert to numpy
        predictions_np = predictions.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()

        # Format response
        results = []
        for i in range(len(predictions_np)):
            result = {
                'prediction': classes[predictions_np[i]],
                'prediction_index': int(predictions_np[i]),
                'confidence': float(probabilities_np[i][predictions_np[i]])
            }

            if return_probs:
                result['probabilities'] = {
                    classes[j]: float(probabilities_np[i][j])
                    for j in range(len(classes))
                }

            results.append(result)

        return jsonify({
            'predictions': results,
            'count': len(results)
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """Evaluate model performance on provided data"""
    if model is None or scaler_stats is None:
        return jsonify({'error': 'Model or scaler not loaded'}), 500

    try:
        data = request.get_json()

        if 'data' not in data or 'labels' not in data:
            return jsonify({'error': 'Missing "data" or "labels" field'}), 400

        input_data = np.array(data['data'])
        true_labels = np.array(data['labels'])

        # Validate input
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        if input_data.shape[0] != len(true_labels):
            return jsonify({
                'error': 'Number of samples and labels must match'
            }), 400

        # Manual normalization
        normalized_data = manual_normalize(input_data)
        X_tensor = torch.FloatTensor(normalized_data).to(device)

        # Predict
        with torch.no_grad():
            logits = model(X_tensor)
            predictions = torch.argmax(logits, dim=1)

        predictions_np = predictions.cpu().numpy()

        # Calculate metrics manually
        accuracy = calculate_accuracy(true_labels, predictions_np)
        metrics = calculate_precision_recall_f1(true_labels, predictions_np, len(classes))
        cm = confusion_matrix_manual(true_labels, predictions_np, len(classes))

        # Per-class metrics
        per_class_metrics = []
        for i in range(len(classes)):
            per_class_metrics.append({
                'class': classes[i],
                'precision': float(metrics['per_class']['precision'][i]),
                'recall': float(metrics['per_class']['recall'][i]),
                'f1_score': float(metrics['per_class']['f1'][i]),
                'support': int(metrics['per_class']['support'][i])
            })

        return jsonify({
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1'])
            },
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'sample_count': len(true_labels)
        })

    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Process batch predictions from uploaded CSV or JSON data"""
    if model is None or scaler_stats is None:
        return jsonify({'error': 'Model or scaler not loaded'}), 500

    try:
        # Check if file upload or JSON
        if request.files and 'file' in request.files:
            file = request.files['file']
            df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF-8")))
            input_data = df.values
        elif request.json and 'data' in request.json:
            input_data = np.array(request.json['data'])
        else:
            return jsonify({'error': 'No data provided'}), 400

        # Validate
        expected_features = len(scaler_stats['mean'])
        if input_data.shape[1] != expected_features:
            return jsonify({
                'error': f'Expected {expected_features} features, got {input_data.shape[1]}'
            }), 400

        # Process in batches to avoid memory issues
        batch_size = 1000
        all_predictions = []

        for i in range(0, len(input_data), batch_size):
            batch = input_data[i:i+batch_size]
            normalized_batch = manual_normalize(batch)
            X_tensor = torch.FloatTensor(normalized_batch).to(device)

            with torch.no_grad():
                logits = model(X_tensor)
                predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy().tolist())

        # Count predictions per class
        pred_counts = {}
        for pred_idx in all_predictions:
            class_name = classes[pred_idx]
            pred_counts[class_name] = pred_counts.get(class_name, 0) + 1

        return jsonify({
            'predictions': [classes[idx] for idx in all_predictions],
            'prediction_counts': pred_counts,
            'total_samples': len(all_predictions)
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analysis/summary', methods=['GET'])
def get_analysis_summary():
    """
    Get summary of the latest file analysis

    Returns high-level statistics including:
    - Total samples analyzed
    - Anomaly count and rate
    - Attack type breakdown
    - Analysis timestamp
    """
    if latest_analysis['results'] is None:
        return jsonify({
            'error': 'No analysis available. Please upload and analyze a file first.',
            'available': False
        }), 404

    results = latest_analysis['results']

    return jsonify({
        'available': True,
        'filename': latest_analysis['filename'],
        'timestamp': latest_analysis['timestamp'],
        'summary': {
            'total_samples': results['analysis']['total_samples'],
            'benign_count': results['analysis']['benign_count'],
            'anomaly_count': results['analysis']['anomaly_count'],
            'anomaly_rate': results['analysis']['anomaly_rate'],
            'severity': results['summary']['severity'],
            'unique_attack_types': results['summary']['unique_attack_types'],
            'most_common_attack': results['summary']['most_common_attack']
        }
    })

@app.route('/analysis/attacks', methods=['GET'])
def get_attack_distribution():
    """
    Get attack type distribution for visualization

    Returns breakdown of all detected attack types with:
    - Attack type name
    - Count of occurrences
    - Percentage of total samples
    - Average confidence score
    """
    if latest_analysis['results'] is None:
        return jsonify({
            'error': 'No analysis available. Please upload and analyze a file first.',
            'available': False
        }), 404

    results = latest_analysis['results']

    # Format for easier visualization
    attack_data = []
    for attack_type, stats in results['attack_breakdown'].items():
        attack_data.append({
            'attack_type': attack_type,
            'count': stats['count'],
            'percentage': stats['percentage'],
            'confidence': results['attack_confidence'].get(attack_type, 0)
        })

    # Sort by count descending
    attack_data.sort(key=lambda x: x['count'], reverse=True)

    return jsonify({
        'available': True,
        'filename': latest_analysis['filename'],
        'timestamp': latest_analysis['timestamp'],
        'attacks': attack_data,
        'total_attack_types': len(attack_data)
    })

@app.route('/analysis/anomalies', methods=['GET'])
def get_anomaly_details():
    """
    Get detailed anomaly information

    Query parameters:
    - limit: Maximum number of anomalies to return (default: 100)
    - min_confidence: Minimum confidence threshold (default: 0.0)
    - attack_type: Filter by specific attack type (optional)

    Returns list of detected anomalies with confidence scores and probabilities
    """
    if latest_analysis['results'] is None:
        return jsonify({
            'error': 'No analysis available. Please upload and analyze a file first.',
            'available': False
        }), 404

    results = latest_analysis['results']

    # Get query parameters
    limit = request.args.get('limit', default=100, type=int)
    min_confidence = request.args.get('min_confidence', default=0.0, type=float)
    attack_type_filter = request.args.get('attack_type', default=None, type=str)

    # Filter anomalies
    anomalies = results['top_anomalies']

    if attack_type_filter:
        anomalies = [a for a in anomalies if a['attack_type'] == attack_type_filter]

    if min_confidence > 0:
        anomalies = [a for a in anomalies if a['confidence'] >= min_confidence]

    # Apply limit
    anomalies = anomalies[:limit]

    return jsonify({
        'available': True,
        'filename': latest_analysis['filename'],
        'timestamp': latest_analysis['timestamp'],
        'anomalies': anomalies,
        'count': len(anomalies),
        'filters': {
            'limit': limit,
            'min_confidence': min_confidence,
            'attack_type': attack_type_filter
        }
    })

@app.route('/analysis/confidence', methods=['GET'])
def get_confidence_metrics():
    """
    Get confidence score metrics for all attack types

    Returns:
    - Average confidence per attack type
    - Overall confidence statistics
    - Confidence distribution data
    """
    if latest_analysis['results'] is None:
        return jsonify({
            'error': 'No analysis available. Please upload and analyze a file first.',
            'available': False
        }), 404

    results = latest_analysis['results']

    # Format confidence data
    confidence_data = []
    all_confidences = []

    for attack_type, avg_confidence in results['attack_confidence'].items():
        confidence_data.append({
            'attack_type': attack_type,
            'average_confidence': avg_confidence,
            'sample_count': results['attack_breakdown'][attack_type]['count']
        })
        all_confidences.append(avg_confidence)

    # Calculate overall statistics
    overall_stats = {}
    if all_confidences:
        overall_stats = {
            'mean': float(np.mean(all_confidences)),
            'median': float(np.median(all_confidences)),
            'std': float(np.std(all_confidences)),
            'min': float(np.min(all_confidences)),
            'max': float(np.max(all_confidences))
        }

    return jsonify({
        'available': True,
        'filename': latest_analysis['filename'],
        'timestamp': latest_analysis['timestamp'],
        'confidence_by_attack': confidence_data,
        'overall_statistics': overall_stats
    })

@app.route('/analysis/timeline', methods=['GET'])
def get_anomaly_timeline():
    """
    Get anomaly detection timeline data for time-series visualization

    Query parameters:
    - window_size: Number of samples per time window (default: 100)

    Returns anomaly counts over sequential windows for trend visualization
    """
    if latest_analysis['results'] is None:
        return jsonify({
            'error': 'No analysis available. Please upload and analyze a file first.',
            'available': False
        }), 404

    results = latest_analysis['results']
    window_size = request.args.get('window_size', default=100, type=int)

    # Get all predictions from stored results
    anomalies = results['top_anomalies']
    total_samples = results['analysis']['total_samples']

    # Create timeline windows
    num_windows = max(1, total_samples // window_size)
    timeline = []

    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, total_samples)

        # Count anomalies in this window
        window_anomalies = [a for a in anomalies if start_idx <= a['sample_index'] < end_idx]

        # Count by attack type
        attack_counts = {}
        for anomaly in window_anomalies:
            attack_type = anomaly['attack_type']
            attack_counts[attack_type] = attack_counts.get(attack_type, 0) + 1

        timeline.append({
            'window': i,
            'sample_range': f"{start_idx}-{end_idx}",
            'anomaly_count': len(window_anomalies),
            'attack_breakdown': attack_counts
        })

    return jsonify({
        'available': True,
        'filename': latest_analysis['filename'],
        'timestamp': latest_analysis['timestamp'],
        'timeline': timeline,
        'window_size': window_size,
        'total_windows': num_windows
    })

@app.route('/analyze_file', methods=['POST'])
def analyze_file():
    """
    Analyze uploaded network traffic file for anomalies and attacks

    Accepts CSV files with network traffic features.
    Returns detailed anomaly analysis including:
    - Total anomalies detected
    - Breakdown by attack type
    - Anomaly rate percentage
    - Individual predictions with confidence scores
    """
    if model is None or scaler_stats is None:
        return jsonify({'error': 'Model or scaler not loaded'}), 500

    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided. Please upload a CSV file.'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are supported'}), 400

        logger.info(f"Analyzing file: {file.filename}")

        # Read CSV file
        try:
            file_content = file.stream.read().decode("UTF-8")
            df = pd.read_csv(io.StringIO(file_content))
        except Exception as e:
            return jsonify({'error': f'Failed to parse CSV file: {str(e)}'}), 400

        # Validate data
        if df.empty:
            return jsonify({'error': 'CSV file is empty'}), 400

        input_data = df.values
        expected_features = len(scaler_stats['mean'])

        if input_data.shape[1] != expected_features:
            return jsonify({
                'error': f'Invalid number of features. Expected {expected_features}, got {input_data.shape[1]}. '
                         f'Please ensure your CSV has the correct feature columns.'
            }), 400

        logger.info(f"Processing {len(input_data)} samples from {file.filename}")

        # Process in batches with detailed predictions
        batch_size = 1000
        all_predictions = []
        all_confidences = []
        all_probabilities = []

        for i in range(0, len(input_data), batch_size):
            batch = input_data[i:i+batch_size]
            normalized_batch = manual_normalize(batch)
            X_tensor = torch.FloatTensor(normalized_batch).to(device)

            with torch.no_grad():
                logits = model(X_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]

            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_confidences.extend(confidences.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

        # Analyze results
        total_samples = len(all_predictions)

        # Count predictions per class
        pred_counts = {}
        for pred_idx in all_predictions:
            class_name = classes[pred_idx]
            pred_counts[class_name] = pred_counts.get(class_name, 0) + 1

        # Calculate anomaly statistics
        benign_count = pred_counts.get('BENIGN', 0)
        anomaly_count = total_samples - benign_count
        anomaly_rate = (anomaly_count / total_samples * 100) if total_samples > 0 else 0

        # Group attack types
        attack_types = {}
        for class_name, count in pred_counts.items():
            if class_name != 'BENIGN':
                attack_types[class_name] = {
                    'count': count,
                    'percentage': (count / total_samples * 100) if total_samples > 0 else 0
                }

        # Sort attack types by count
        attack_types = dict(sorted(attack_types.items(), key=lambda x: x[1]['count'], reverse=True))

        # Find high-confidence anomalies (for detailed view)
        anomaly_details = []
        for i, (pred_idx, confidence) in enumerate(zip(all_predictions, all_confidences)):
            class_name = classes[pred_idx]
            if class_name != 'BENIGN':
                anomaly_details.append({
                    'sample_index': i,
                    'attack_type': class_name,
                    'confidence': float(confidence),
                    'probabilities': {
                        classes[j]: float(all_probabilities[i][j])
                        for j in range(len(classes))
                    }
                })

        # Limit detailed anomalies to top 100 for response size
        top_anomalies = sorted(anomaly_details, key=lambda x: x['confidence'], reverse=True)[:100]

        # Calculate average confidence per attack type
        attack_confidence = {}
        for class_name in attack_types.keys():
            confidences = [all_confidences[i] for i, pred_idx in enumerate(all_predictions)
                          if classes[pred_idx] == class_name]
            if confidences:
                attack_confidence[class_name] = float(np.mean(confidences))

        logger.info(f"Analysis complete: {anomaly_count}/{total_samples} anomalies detected ({anomaly_rate:.2f}%)")

        # Prepare response
        response_data = {
            'success': True,
            'filename': file.filename,
            'analysis': {
                'total_samples': total_samples,
                'benign_count': benign_count,
                'anomaly_count': anomaly_count,
                'anomaly_rate': round(anomaly_rate, 2)
            },
            'attack_breakdown': attack_types,
            'attack_confidence': attack_confidence,
            'top_anomalies': top_anomalies,
            'summary': {
                'most_common_attack': max(attack_types.items(), key=lambda x: x[1]['count'])[0] if attack_types else None,
                'unique_attack_types': len(attack_types),
                'severity': 'HIGH' if anomaly_rate > 50 else 'MEDIUM' if anomaly_rate > 20 else 'LOW'
            }
        }

        # Store results globally for visualization endpoints
        global latest_analysis
        latest_analysis['results'] = response_data
        latest_analysis['timestamp'] = datetime.now().isoformat()
        latest_analysis['filename'] = file.filename

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"File analysis error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
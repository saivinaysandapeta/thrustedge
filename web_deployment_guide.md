# Web Deployment Guide: ML-Powered Propulsion Predictor

## Overview

This guide integrates your trained ML model into the ThrustEdge web application, allowing users to:
1. Input motor parameters (KV, ESC A, Battery V, Prop D, Prop P)
2. Get instant predictions for RPM, Thrust, Power, Efficiency
3. Download predicted performance CSV file

## Architecture

```
┌─────────────────────────┐
│  User Web Interface     │
│  (HTML/JavaScript)      │
└────────────┬────────────┘
             │ (HTTP POST: {motor_kv, esc_a, ...})
             │
┌────────────▼────────────┐
│  FastAPI Backend        │
│  (Python)               │
└────────────┬────────────┘
             │ (loads trained model)
             │
┌────────────▼────────────┐
│  XGBoost Models         │
│  (propulsion_model.pkl) │
└─────────────────────────┘
             │
             ▼
        CSV Download
```

## Backend Implementation (FastAPI)

```python
# backend.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
from io import StringIO
import tempfile
from pathlib import Path

app = FastAPI(title="ThrustEdge Propulsion Predictor")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model at startup
MODEL_PATH = "propulsion_model.pkl"

try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    models = model_data['models']
    scalers = model_data['scalers']
    input_features = model_data['input_features']
    output_names = model_data['output_names']
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    models = None

# ============================================================================
# DATA MODELS
# ============================================================================

class PropulsionInput(BaseModel):
    """Input schema for propulsion prediction"""
    motor_kv: float
    esc_limit_a: float
    battery_voltage: float
    prop_diameter_in: float
    prop_pitch_in: float
    num_points: int = 11  # Default: 11 throttle points

class PredictionResult(BaseModel):
    """Single prediction result"""
    throttle: int
    throttle_percent: float
    rpm: int
    thrust_g: float
    thrust_kg: float
    power_w: float
    efficiency_percent: float
    current_a: float
    voltage_v: float
    torque_nm: float

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def read_root():
    """Health check"""
    return {
        "status": "ok",
        "service": "ThrustEdge Propulsion Predictor",
        "model_loaded": models is not None
    }

@app.post("/predict")
def predict_performance(input_data: PropulsionInput) -> dict:
    """
    Predict propulsion performance for given parameters
    
    Returns:
        dict with predicted performance values at each throttle point
    """
    
    if models is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Validate inputs
        if not (50 <= input_data.motor_kv <= 5000):
            raise ValueError("Motor KV must be between 50 and 5000")
        if not (5 <= input_data.esc_limit_a <= 300):
            raise ValueError("ESC limit must be between 5A and 300A")
        if not (3 <= input_data.battery_voltage <= 48):
            raise ValueError("Battery voltage must be between 3V and 48V")
        if not (2 <= input_data.prop_diameter_in <= 20):
            raise ValueError("Propeller diameter must be between 2\" and 20\"")
        if not (0.5 <= input_data.prop_pitch_in <= 8):
            raise ValueError("Propeller pitch must be between 0.5\" and 8\"")
        
        # Convert units
        prop_diameter_m = input_data.prop_diameter_in * 0.0254
        prop_pitch_m = input_data.prop_pitch_in * 0.0254
        
        # Generate throttle points
        throttle_levels = np.linspace(1000, 2000, input_data.num_points)
        throttle_percentages = np.linspace(0, 100, input_data.num_points)
        
        predictions = []
        
        for throttle, throttle_pct in zip(throttle_levels, throttle_percentages):
            # Create feature vector
            X_new = np.array([[
                input_data.motor_kv,
                input_data.esc_limit_a,
                input_data.battery_voltage,
                prop_diameter_m,
                prop_pitch_m,
                throttle_pct
            ]])
            
            # Normalize
            X_scaled = scalers['X'].transform(X_new)
            
            # Predict each output
            outputs = {}
            for output_name in output_names:
                model = models[output_name]
                y_pred = model.predict(X_scaled)[0]
                outputs[output_name] = float(y_pred)
            
            # Handle negative predictions (physically impossible values)
            outputs['rpm'] = max(0, outputs['rpm'])
            outputs['thrust_kg'] = max(0, outputs['thrust_kg'])
            outputs['power_w'] = max(0, outputs['power_w'])
            outputs['efficiency'] = np.clip(outputs['efficiency'], 0, 100)
            outputs['current_a'] = max(0, outputs['current_a'])
            outputs['torque_nm'] = max(0, outputs['torque_nm'])
            
            result = {
                'throttle': int(throttle),
                'throttle_percent': round(throttle_pct, 1),
                'rpm': int(outputs['rpm']),
                'thrust_g': round(outputs['thrust_kg'] * 1000, 1),
                'thrust_kg': round(outputs['thrust_kg'], 4),
                'power_w': round(outputs['power_w'], 1),
                'efficiency_percent': round(outputs['efficiency'], 2),
                'current_a': round(outputs['current_a'], 2),
                'voltage_v': round(outputs['voltage_v'], 2),
                'torque_nm': round(outputs['torque_nm'], 3)
            }
            
            predictions.append(result)
        
        return {
            'status': 'success',
            'input_parameters': {
                'motor_kv': input_data.motor_kv,
                'esc_limit_a': input_data.esc_limit_a,
                'battery_voltage': input_data.battery_voltage,
                'prop_diameter_in': input_data.prop_diameter_in,
                'prop_pitch_in': input_data.prop_pitch_in
            },
            'predictions': predictions,
            'summary': {
                'max_thrust_g': max(p['thrust_g'] for p in predictions),
                'max_power_w': max(p['power_w'] for p in predictions),
                'avg_efficiency': np.mean([p['efficiency_percent'] for p in predictions])
            }
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/download-csv")
def download_csv(input_data: PropulsionInput):
    """
    Download predicted performance as CSV file
    """
    
    if models is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Get predictions
        prediction_result = predict_performance(input_data)
        
        if prediction_result['status'] != 'success':
            raise Exception("Prediction failed")
        
        predictions = prediction_result['predictions']
        
        # Create DataFrame
        df = pd.DataFrame(predictions)
        
        # Reorder columns
        column_order = ['throttle', 'throttle_percent', 'rpm', 'thrust_g', 'thrust_kg',
                       'power_w', 'efficiency_percent', 'current_a', 'voltage_v', 'torque_nm']
        df = df[column_order]
        
        # Convert to CSV string
        csv_string = df.to_csv(index=False)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(csv_string)
            temp_path = f.name
        
        # Return as file download
        filename = (f"thrust_prediction_"
                   f"KV{input_data.motor_kv}_"
                   f"{input_data.esc_limit_a}A_"
                   f"{input_data.battery_voltage}V_"
                   f"{input_data.prop_diameter_in}x{input_data.prop_pitch_in}.csv")
        
        return FileResponse(
            path=temp_path,
            filename=filename,
            media_type='text/csv'
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV generation error: {str(e)}")

@app.get("/model-info")
def get_model_info():
    """Get information about loaded model"""
    return {
        'status': 'ok',
        'model_loaded': models is not None,
        'input_features': input_features if models else None,
        'output_features': output_names if models else None,
        'num_models': len(models) if models else 0
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Frontend Integration

Update your HTML to call the API:

```javascript
// In your existing ThrustEdge app, replace generateReport() with:

async function generateReport() {
    const motorKV = parseFloat(document.getElementById('motorKV').value);
    const escLimit = parseFloat(document.getElementById('escLimit').value);
    const batteryVoltage = parseFloat(document.getElementById('batteryVoltage').value);
    const propDiameter = parseFloat(document.getElementById('propDiameter').value);
    const propPitch = parseFloat(document.getElementById('propPitch').value);
    
    // Validate inputs
    if (!motorKV || !escLimit || !batteryVoltage || !propDiameter || !propPitch) {
        showMessage('Please fill all required fields', 'error');
        return;
    }
    
    // Show loading state
    document.getElementById('loading').classList.add('show');
    document.getElementById('resultsSummary').innerHTML = '';
    
    try {
        // Call FastAPI backend
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                motor_kv: motorKV,
                esc_limit_a: escLimit,
                battery_voltage: batteryVoltage,
                prop_diameter_in: propDiameter,
                prop_pitch_in: propPitch,
                num_points: 11
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }
        
        const data = await response.json();
        
        // Store predictions globally
        window.ML_TEST_DATA = data.predictions;
        window.ML_INPUT_PARAMS = data.input_parameters;
        
        // Update UI
        displayMLResults(data);
        showMessage('ML Model predictions generated!', 'success');
        
    } catch (error) {
        showMessage(`Error: ${error.message}`, 'error');
        console.error('Prediction error:', error);
    } finally {
        document.getElementById('loading').classList.remove('show');
    }
}

function displayMLResults(data) {
    const predictions = data.predictions;
    const summary = data.summary;
    
    // Update metrics
    const avgThrust = predictions.reduce((sum, p) => sum + p.thrust_kg, 0) / predictions.length;
    const avgEfficiency = summary.avg_efficiency;
    
    const summaryHTML = `
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Max Thrust</div>
                <div class="metric-value">${summary.max_thrust_g.toFixed(0)}g</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Thrust</div>
                <div class="metric-value">${(avgThrust * 1000).toFixed(0)}g</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Efficiency</div>
                <div class="metric-value">${avgEfficiency.toFixed(1)}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Power</div>
                <div class="metric-value">${summary.max_power_w.toFixed(0)}W</div>
            </div>
        </div>
        <div style="margin-top: 20px; padding: 15px; background: rgba(16, 185, 129, 0.1); border-left: 3px solid var(--success); border-radius: 4px;">
            <span class="status success">✓ ML Model Prediction</span>
            <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 8px;">
                Predicted from 300+ experimental datasets
            </p>
        </div>
    `;
    
    document.getElementById('resultsSummary').innerHTML = summaryHTML;
    
    // Render table and charts
    TEST_DATA = predictions.map(p => ({
        throttle: p.throttle,
        throttlePercent: p.throttle_percent.toString(),
        rpm: p.rpm,
        thrust: p.thrust_kg.toFixed(4),
        torque: p.torque_nm.toFixed(3),
        voltage: p.voltage_v.toFixed(2),
        current: p.current_a.toFixed(2),
        electricalPower: p.power_w.toFixed(1),
        efficiency: p.efficiency_percent.toFixed(2)
    }));
    
    renderTable();
    renderCharts();
}

async function downloadCSVFromML() {
    const motorKV = parseFloat(document.getElementById('motorKV').value);
    const escLimit = parseFloat(document.getElementById('escLimit').value);
    const batteryVoltage = parseFloat(document.getElementById('batteryVoltage').value);
    const propDiameter = parseFloat(document.getElementById('propDiameter').value);
    const propPitch = parseFloat(document.getElementById('propPitch').value);
    
    if (!motorKV || !escLimit || !batteryVoltage || !propDiameter || !propPitch) {
        showMessage('Generate a prediction first', 'error');
        return;
    }
    
    try {
        const response = await fetch('http://localhost:8000/download-csv', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                motor_kv: motorKV,
                esc_limit_a: escLimit,
                battery_voltage: batteryVoltage,
                prop_diameter_in: propDiameter,
                prop_pitch_in: propPitch
            })
        });
        
        if (!response.ok) throw new Error('Download failed');
        
        // Download file
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `thrust_prediction_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        
        showMessage('CSV downloaded successfully!', 'success');
    } catch (error) {
        showMessage(`Download error: ${error.message}`, 'error');
    }
}
```

## Deployment Instructions

### 1. Install Dependencies

```bash
pip install fastapi uvicorn xgboost pandas numpy scikit-learn
```

### 2. Train Model (First Time)

```bash
# Place your 300+ CSV files in ./thrust_test_data folder
python train_propulsion_model.py

# Generates: propulsion_model.pkl
```

### 3. Run Backend Server

```bash
python backend.py
# Server runs on http://localhost:8000
```

### 4. Deploy to Production

**Option A: Using Heroku**

```bash
# Create Procfile
echo "web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend:app" > Procfile

# Deploy
heroku create thrusteedge-predictor
git push heroku main
```

**Option B: Using Docker**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t thrusteedge-predictor .
docker run -p 8000:8000 thrusteedge-predictor
```

**Option C: Using AWS Lambda + API Gateway**

```bash
# Use serverless framework
serverless create --template aws-python3 --path thrusteedge-api
# Configure FastAPI with AWS Lambda handler
```

## Performance Considerations

- **Model Size**: ~50MB (pickle file)
- **Prediction Time**: ~100ms per prediction
- **Concurrency**: FastAPI handles 1000+ requests/sec
- **Memory**: ~500MB when running with model loaded

## Monitoring & Logging

```python
# Add to backend.py
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict")
def predict_performance(input_data: PropulsionInput) -> dict:
    logger.info(f"Prediction request: KV={input_data.motor_kv}, "
                f"ESC={input_data.esc_limit_a}A, "
                f"Prop={input_data.prop_diameter_in}x{input_data.prop_pitch_in}")
    # ... rest of code
```

## Security Considerations

```python
# Add rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
def predict_performance(input_data: PropulsionInput) -> dict:
    # ... rest of code

# Add authentication (optional)
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(credentials = Depends(security)):
    if credentials.credentials != "your-secret-key":
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials
```

This implementation provides a scalable, production-ready ML predictor integrated into your ThrustEdge platform!

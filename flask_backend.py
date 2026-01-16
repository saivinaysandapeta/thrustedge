
"""
Flask Backend for ML-Powered Propulsion Test Generator
======================================================

API Endpoints:
  POST /api/predict - Generate test report for new configuration
  GET /api/config-options - Get available options
  POST /api/batch - Generate multiple reports
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import io
import json

# Import the ML pipeline
from propulsion_ml_backend import PropulsionModelPipeline

app = Flask(__name__)
CORS(app)  # Enable CORS for web requests

# Initialize ML pipeline
try:
    pipeline = PropulsionModelPipeline(model_dir='./propulsion_models')
    pipeline.load_models()
    print("✓ ML models loaded successfully")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    pipeline = None

# Configuration options
CONFIG_OPTIONS = {
    "motors": [
        {"kv": 450, "mfg": "T-Motor", "class": "Heavy Lift"},
        {"kv": 900, "mfg": "MWD", "class": "Mapping"},
        {"kv": 1200, "mfg": "T-Motor", "class": "Medium"},
        {"kv": 2850, "mfg": "T-Motor", "class": "Racing"},
        {"kv": 3000, "mfg": "DJI", "class": "FPV"},
    ],
    "batteries": [
        {"voltage": 7.4, "cells": 2, "capacity": "1300mAh"},
        {"voltage": 11.1, "cells": 3, "capacity": "2200mAh"},
        {"voltage": 14.8, "cells": 4, "capacity": "3000mAh"},
        {"voltage": 18.5, "cells": 5, "capacity": "5000mAh"},
        {"voltage": 22.2, "cells": 6, "capacity": "8000mAh"},
    ],
    "propellers": [
        {"diameter": 3, "pitch": 2, "mfg": "Master Airscrew (MAS)"},
        {"diameter": 4, "pitch": 3, "mfg": "APC"},
        {"diameter": 6, "pitch": 4, "mfg": "Master Airscrew (MAS)"},
        {"diameter": 8, "pitch": 5, "mfg": "APC"},
        {"diameter": 10, "pitch": 5, "mfg": "APC"},
        {"diameter": 12, "pitch": 4, "mfg": "Master Airscrew (MAS)"},
        {"diameter": 15, "pitch": 5, "mfg": "APC"},
    ],
    "escs": [
        {"current": 10, "mfg": "Hobbywing"},
        {"current": 20, "mfg": "Hobbywing"},
        {"current": 30, "mfg": "Hobbywing"},
        {"current": 40, "mfg": "Hobbywing"},
        {"current": 60, "mfg": "T-Motor"},
        {"current": 80, "mfg": "T-Motor"},
    ]
}

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": pipeline is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/config-options', methods=['GET'])
def get_config_options():
    """Get available configuration options"""
    return jsonify(CONFIG_OPTIONS)

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Generate test report for given motor/ESC/battery/propeller config

    Request JSON:
    {
        "motor_kv": 2850,
        "esc_limit_a": 30,
        "battery_voltage_v": 11.1,
        "prop_diameter_in": 6,
        "prop_pitch_in": 4,
        "motor_mfg": "T-Motor",
        "esc_mfg": "Hobbywing",
        "prop_mfg": "Master Airscrew",
        "throttle_start": 1000,
        "throttle_end": 2000,
        "throttle_step": 50
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['motor_kv', 'esc_limit_a', 'battery_voltage_v', 
                          'prop_diameter_in', 'prop_pitch_in']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        if not pipeline:
            return jsonify({"error": "ML models not loaded"}), 503

        # Generate throttle levels
        throttle_start = data.get('throttle_start', 1000)
        throttle_end = data.get('throttle_end', 2000)
        throttle_step = data.get('throttle_step', 50)
        throttle_levels = list(range(int(throttle_start), int(throttle_end) + 1, int(throttle_step)))

        # Generate report
        report = pipeline.generate_test_report(
            motor_kv=float(data['motor_kv']),
            esc_limit_a=float(data['esc_limit_a']),
            battery_voltage_v=float(data['battery_voltage_v']),
            prop_diameter_in=float(data['prop_diameter_in']),
            prop_pitch_in=float(data['prop_pitch_in']),
            prop_mfg=data.get('prop_mfg', 'Unknown'),
            motor_mfg=data.get('motor_mfg', 'Unknown'),
            esc_mfg=data.get('esc_mfg', 'Unknown'),
            throttle_levels=throttle_levels
        )

        # Convert to CSV
        csv_buffer = io.StringIO()
        report.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"thrust_test_{data['motor_kv']}kv_{data['prop_diameter_in']}x{data['prop_pitch_in']}_{timestamp}.csv"

        # Return as JSON with CSV content (for preview)
        return jsonify({
            "status": "success",
            "filename": filename,
            "rows": len(report),
            "data": report.to_dict(orient='records'),
            "summary": {
                "max_thrust_kgf": float(report['Thrust (kgf)'].max()),
                "max_rpm": int(report['Rotation speed (rpm)'].max()),
                "max_power_w": float(report['Electrical power (W)'].max()),
                "max_current_a": float(report['Current (A)'].max()),
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/download', methods=['POST'])
def predict_download():
    """
    Generate and download test report as CSV file
    """
    try:
        data = request.get_json()

        # Generate report (same as /api/predict)
        throttle_levels = list(range(1000, 2050, 50))
        report = pipeline.generate_test_report(
            motor_kv=float(data['motor_kv']),
            esc_limit_a=float(data['esc_limit_a']),
            battery_voltage_v=float(data['battery_voltage_v']),
            prop_diameter_in=float(data['prop_diameter_in']),
            prop_pitch_in=float(data['prop_pitch_in']),
            prop_mfg=data.get('prop_mfg', 'Unknown'),
            motor_mfg=data.get('motor_mfg', 'Unknown'),
            esc_mfg=data.get('esc_mfg', 'Unknown'),
            throttle_levels=throttle_levels
        )

        # Create CSV file
        csv_buffer = io.BytesIO()
        report.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        filename = f"thrust_test_{data['motor_kv']}kv_{data['prop_diameter_in']}x{data['prop_pitch_in']}.csv"

        return send_file(
            csv_buffer,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Generate reports for multiple configurations

    Request JSON:
    {
        "configurations": [
            {"motor_kv": 2850, "esc_limit_a": 30, ...},
            {"motor_kv": 1200, "esc_limit_a": 40, ...},
        ]
    }
    """
    try:
        data = request.get_json()
        configs = data.get('configurations', [])

        results = []
        for i, config in enumerate(configs):
            try:
                throttle_levels = list(range(1000, 2050, 50))
                report = pipeline.generate_test_report(
                    motor_kv=float(config['motor_kv']),
                    esc_limit_a=float(config['esc_limit_a']),
                    battery_voltage_v=float(config['battery_voltage_v']),
                    prop_diameter_in=float(config['prop_diameter_in']),
                    prop_pitch_in=float(config['prop_pitch_in']),
                    prop_mfg=config.get('prop_mfg', 'Unknown'),
                    motor_mfg=config.get('motor_mfg', 'Unknown'),
                    esc_mfg=config.get('esc_mfg', 'Unknown'),
                    throttle_levels=throttle_levels
                )

                results.append({
                    "index": i,
                    "status": "success",
                    "max_thrust": float(report['Thrust (kgf)'].max()),
                    "max_power": float(report['Electrical power (W)'].max()),
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "status": "error",
                    "error": str(e)
                })

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about trained models"""
    if not pipeline:
        return jsonify({"error": "Models not loaded"}), 503

    return jsonify({
        "models": list(pipeline.models.keys()),
        "features": pipeline.feature_names,
        "scaler_mean": pipeline.scaler.mean_.tolist() if pipeline.scaler else None,
        "scaler_std": pipeline.scaler.scale_.tolist() if pipeline.scaler else None,
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("STARTING PROPULSION ML API SERVER")
    print("="*80)
    print("\nAPI Endpoints:")
    print("  GET  /api/health              - Health check")
    print("  GET  /api/config-options      - Available configurations")
    print("  POST /api/predict             - Generate report (JSON response)")
    print("  POST /api/predict/download    - Generate report (CSV download)")
    print("  POST /api/batch-predict       - Multiple reports")
    print("  GET  /api/model-info          - Model information")
    print("\nServer running on http://localhost:5000")
    print("="*80 + "\n")

    app.run(debug=True, port=5000)

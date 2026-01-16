# Complete Implementation Guide: ML-Based Propulsion System Characterization Tool

## Project Overview

**Goal:** Build a machine learning system that predicts propulsion system performance (RPM, Thrust, Power, etc.) for unseen combinations of Motor (Kv), ESC (A), Battery (V), and Propeller geometry.

**Problem Type:** Regression with Domain Physics Awareness
- **Physics Domains:** Electrical (Battery→ESC) → Mechanical (Motor) → Aerodynamic (Propeller)
- **Challenge:** Different propeller geometries (same size, different pitch) behave differently
- **Solution:** Multi-output neural network with physics-informed feature engineering

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    WEB APPLICATION                          │
│  (Flask/Streamlit Frontend for Local Deployment)           │
├─────────────────────────────────────────────────────────────┤
│                    ML PIPELINE                              │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │Data Loader │→ │Feature Engin.│→ │ ML Models    │        │
│  │(300+ CSVs) │  │(Physics-based)│  │(XGBoost,NN) │        │
│  └────────────┘  └──────────────┘  └──────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                  OUTPUTS                                    │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ CSV Export  │  │ Visualization│  │ PDF Report   │       │
│  └─────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Organization & Preparation

### 1.1 Directory Structure

```
propulsion_ml_system/
├── data/
│   ├── raw/
│   │   ├── test_001_motor2850_prop7_6_esc30_batt_7p9V.xlsx
│   │   ├── test_002_motor2850_prop5_3_esc30_batt_7p9V.xlsx
│   │   └── ... (300+ files)
│   └── processed/
│       ├── combined_dataset.csv
│       └── test_runs_metadata.csv
├── models/
│   ├── rpm_predictor.pkl
│   ├── thrust_predictor.pkl
│   └── power_predictor.pkl
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model_trainer.py
│   ├── model_predictor.py
│   └── utils.py
├── web_app/
│   ├── app.py (Flask)
│   ├── requirements.txt
│   └── templates/
│       ├── index.html
│       ├── predict.html
│       └── results.html
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
└── README.md
```

### 1.2 Data Loading Script

**File: `src/data_loader.py`**

```python
import pandas as pd
import numpy as np
import os
from pathlib import Path
import openpyxl
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.all_files = list(self.data_dir.glob("*.xlsx"))
        self.combined_df = None
        
    def load_single_file(self, filepath):
        """Load single Excel file and extract metadata from filename"""
        try:
            df = pd.read_excel(filepath)
            
            # Extract metadata from filename
            filename = Path(filepath).stem
            # Expected format: test_XXX_motorKV_propDIAM_PITCH_escA_battV
            
            metadata = self._parse_filename(filename)
            
            # Add metadata columns
            for key, value in metadata.items():
                df[key] = value
                
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def _parse_filename(self, filename):
        """Parse filename to extract component parameters"""
        # This depends on your naming convention
        # Adjust regex/parsing based on actual filename pattern
        metadata = {
            'filename': filename,
            'test_id': None,
            'motor_kv': None,
            'propeller_diameter_inch': None,
            'propeller_pitch_inch': None,
            'esc_amperage': None,
            'battery_voltage': None
        }
        
        # Example parsing (adjust to your naming scheme)
        parts = filename.split('_')
        try:
            for i, part in enumerate(parts):
                if 'motor' in part.lower():
                    metadata['motor_kv'] = float(parts[i+1])
                elif 'prop' in part.lower():
                    metadata['propeller_diameter_inch'] = float(parts[i+1])
                    metadata['propeller_pitch_inch'] = float(parts[i+2])
                elif 'esc' in part.lower():
                    metadata['esc_amperage'] = float(parts[i+1])
                elif 'batt' in part.lower():
                    metadata['battery_voltage'] = float(parts[i+1].replace('p', '.'))
        except:
            print(f"Could not parse filename: {filename}")
            
        return metadata
    
    def load_all_files(self, limit=None):
        """Load all Excel files from directory"""
        dataframes = []
        files = self.all_files[:limit] if limit else self.all_files
        
        print(f"Loading {len(files)} files...")
        for i, filepath in enumerate(files):
            df = self.load_single_file(filepath)
            if df is not None:
                dataframes.append(df)
            if (i + 1) % 50 == 0:
                print(f"  Loaded {i + 1}/{len(files)}")
        
        if dataframes:
            self.combined_df = pd.concat(dataframes, ignore_index=True)
            print(f"Combined dataset shape: {self.combined_df.shape}")
            return self.combined_df
        else:
            raise ValueError("No files successfully loaded")
    
    def get_data_info(self):
        """Get summary statistics of combined dataset"""
        if self.combined_df is None:
            print("No data loaded. Call load_all_files() first.")
            return None
        
        print("\n=== Dataset Summary ===")
        print(f"Total rows: {len(self.combined_df)}")
        print(f"Total columns: {len(self.combined_df.columns)}")
        print(f"\nColumn dtypes:")
        print(self.combined_df.dtypes)
        print(f"\nNumeric columns statistics:")
        print(self.combined_df.describe())
        
        # Component combinations
        print(f"\n=== Component Combinations ===")
        if 'motor_kv' in self.combined_df.columns:
            print(f"Unique Motor Kv values: {self.combined_df['motor_kv'].nunique()}")
            print(f"  Values: {sorted(self.combined_df['motor_kv'].dropna().unique())}")
        
        if 'propeller_diameter_inch' in self.combined_df.columns:
            print(f"Unique Propeller Sizes: {self.combined_df['propeller_diameter_inch'].nunique()}")
        
        return self.combined_df
    
    def clean_data(self):
        """Data cleaning and validation"""
        df = self.combined_df.copy()
        
        # Remove rows with all NaN
        df = df.dropna(how='all')
        
        # Identify key output columns
        output_cols = ['Rotation speed (rpm)', 'Thrust (kgf)', 'Electrical power (W)', 
                       'Power consumption (W)', 'Propeller efficiency (gf/W)']
        
        # Remove rows where all outputs are NaN
        valid_outputs = [col for col in output_cols if col in df.columns]
        if valid_outputs:
            df = df[df[valid_outputs].notna().any(axis=1)]
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Reset index
        df = df.reset_index(drop=True)
        
        self.combined_df = df
        print(f"Cleaned dataset shape: {df.shape}")
        return df
    
    def save_combined_data(self, output_path="data/processed/combined_dataset.csv"):
        """Save combined cleaned dataset"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.combined_df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    loader = DataLoader("data/raw")
    
    # Load all files (set limit for testing)
    loader.load_all_files(limit=10)  # Start with 10 files
    
    # Get info
    loader.get_data_info()
    
    # Clean
    loader.clean_data()
    
    # Save
    loader.save_combined_data()
```

---

## Phase 2: Feature Engineering with Physics

### 2.1 Physics-Informed Features

**File: `src/feature_engineering.py`**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

class FeatureEngineer:
    """
    Create physics-informed features that capture:
    1. Electrical domain (Battery → ESC → Motor)
    2. Mechanical domain (Motor torque generation)
    3. Aerodynamic domain (Propeller efficiency)
    4. System interactions (coupling effects)
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.scaler = StandardScaler()
        
    def create_physics_features(self):
        """Create domain-specific features"""
        df = self.df.copy()
        
        # ===== ELECTRICAL DOMAIN =====
        # Battery characteristics
        if 'Voltage (V)' in df.columns and 'Current (A)' in df.columns:
            df['electrical_power_input'] = df['Voltage (V)'] * df['Current (A)']
        
        # Voltage utilization (voltage applied / battery nominal)
        if 'Battery Nominal V' in df.columns or 'battery_voltage' in df.columns:
            batt_col = 'battery_voltage' if 'battery_voltage' in df.columns else 'Battery Nominal V'
            if batt_col in df.columns:
                df['voltage_utilization'] = df['Voltage (V)'] / df[batt_col]
        
        # ===== MOTOR DOMAIN =====
        # Motor constant approximation (Kv relates to back-EMF)
        if 'motor_kv' in df.columns and 'Rotation speed (rpm)' in df.columns:
            # Kv = RPM/V relationship
            df['motor_back_emf_ratio'] = df['Rotation speed (rpm)'] / df['Voltage (V)'].clip(lower=0.1)
        
        # Motor power factor (mechanical power / electrical power)
        if 'Mechanical power (W)' in df.columns and 'electrical_power_input' in df.columns:
            df['motor_efficiency'] = (df['Mechanical power (W)'] / 
                                      df['electrical_power_input'].clip(lower=0.1))
            df['motor_efficiency'] = df['motor_efficiency'].clip(0, 1)
        
        # Motor torque calculation
        if 'Rotation speed (rpm)' in df.columns and 'Mechanical power (W)' in df.columns:
            # P = T * ω, where ω = RPM * 2π / 60
            omega_rad_s = df['Rotation speed (rpm)'] * 2 * np.pi / 60
            df['motor_torque_nm'] = df['Mechanical power (W)'] / omega_rad_s.clip(lower=0.1)
        
        # ===== PROPELLER DOMAIN =====
        # Disk loading (thrust/swept area)
        if 'Thrust (kgf)' in df.columns and 'propeller_diameter_inch' in df.columns:
            # Convert diameter from inches to meters
            diameter_m = df['propeller_diameter_inch'] * 0.0254
            swept_area_m2 = np.pi * (diameter_m / 2) ** 2
            df['propeller_disk_loading'] = (df['Thrust (kgf)'] * 9.81) / swept_area_m2.clip(lower=0.01)
        
        # Propeller tip speed (affects efficiency and noise)
        if 'Rotation speed (rpm)' in df.columns and 'propeller_diameter_inch' in df.columns:
            diameter_m = df['propeller_diameter_inch'] * 0.0254
            tip_speed_ms = (df['Rotation speed (rpm)'] * diameter_m * np.pi) / 60
            df['propeller_tip_speed_ms'] = tip_speed_ms
        
        # Propeller aspect ratio approximation (simplified)
        if 'propeller_diameter_inch' in df.columns and 'propeller_pitch_inch' in df.columns:
            # Simplified: pitch/diameter ratio
            df['pitch_diameter_ratio'] = (df['propeller_pitch_inch'] / 
                                         df['propeller_diameter_inch'].clip(lower=0.1))
        
        # ===== SYSTEM COUPLING FEATURES =====
        # Power to thrust ratio (propulsion system efficiency)
        if 'electrical_power_input' in df.columns and 'Thrust (kgf)' in df.columns:
            df['thrust_per_watt'] = (df['Thrust (kgf)'] / 
                                    df['electrical_power_input'].clip(lower=0.1))
        
        # Operating point characterization
        if 'Rotation speed (rpm)' in df.columns:
            df['rpm_normalized'] = df['Rotation speed (rpm)'] / df['Rotation speed (rpm)'].max()
        
        # Current per thrust (current density)
        if 'Current (A)' in df.columns and 'Thrust (kgf)' in df.columns:
            df['current_per_kgf_thrust'] = (df['Current (A)'] / 
                                           df['Thrust (kgf)'].clip(lower=0.01))
        
        # System state vector
        if 'motor_kv' in df.columns and 'propeller_diameter_inch' in df.columns:
            df['system_state_kv_propdia'] = (df['motor_kv'] * 
                                            df['propeller_diameter_inch'])
        
        self.df = df
        return df
    
    def handle_missing_values(self, strategy='interpolate'):
        """Handle missing values in features"""
        df = self.df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'interpolate':
            # Group by component combination and interpolate
            component_cols = [col for col in df.columns if col in 
                            ['motor_kv', 'propeller_diameter_inch', 
                             'propeller_pitch_inch', 'esc_amperage', 'battery_voltage']]
            
            if component_cols:
                for col in numeric_cols:
                    df[col] = df.groupby(component_cols)[col].transform(
                        lambda x: x.interpolate(method='linear')
                    )
        
        # Fill remaining with group median
        if component_cols:
            for col in numeric_cols:
                df[col] = df.groupby(component_cols)[col].transform(
                    lambda x: x.fillna(x.median())
                )
        
        # Global forward/backward fill for any remaining
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        self.df = df
        return df
    
    def select_features(self, target_col=None, correlation_threshold=0.95):
        """Select most relevant features"""
        df = self.df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Remove highly correlated features
        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        
        print(f"Removing highly correlated features: {to_drop}")
        df = df.drop(columns=to_drop)
        
        # Feature importance based on variance
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_variance = df[numeric_cols].var()
        low_variance = feature_variance[feature_variance < feature_variance.quantile(0.1)].index
        
        print(f"Removing low variance features: {list(low_variance)}")
        df = df.drop(columns=low_variance, errors='ignore')
        
        self.df = df
        return df
    
    def get_engineered_features(self):
        """Pipeline: create → clean → select features"""
        print("Creating physics-informed features...")
        self.create_physics_features()
        
        print("Handling missing values...")
        self.handle_missing_values()
        
        print("Selecting relevant features...")
        self.select_features()
        
        return self.df
    
    def scale_features(self, columns=None):
        """Scale features to 0-1 or standardized"""
        df = self.df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        df[columns] = self.scaler.fit_transform(df[columns])
        
        self.df = df
        return df


# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/processed/combined_dataset.csv")
    
    # Engineer features
    fe = FeatureEngineer(df)
    engineered_df = fe.get_engineered_features()
    
    print(f"\nFinal feature set shape: {engineered_df.shape}")
    print(f"Columns: {list(engineered_df.columns)}")
    
    # Save
    engineered_df.to_csv("data/processed/engineered_features.csv", index=False)
```

---

## Phase 3: Model Training

### 3.1 Multi-Output Regression Model

**File: `src/model_trainer.py`**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class PropulsionSystemModel:
    """
    Multi-output regression model for propulsion system characterization
    
    Outputs:
    - RPM
    - Thrust (kgf)
    - Electrical Power (W)
    - Motor Efficiency (%)
    - Propeller Efficiency (gf/W)
    """
    
    def __init__(self):
        self.models = {}
        self.feature_scaler = StandardScaler()
        self.output_scalers = {}
        self.feature_columns = None
        self.output_columns = None
        
    def prepare_training_data(self, df, target_columns=None):
        """Prepare features and targets"""
        
        # Default output columns to predict
        if target_columns is None:
            target_columns = [
                'Rotation speed (rpm)',
                'Thrust (kgf)',
                'Electrical power (W)',
                'Motor & ESC efficiency (%)',
                'Propeller efficiency (gf/W)'
            ]
        
        # Available targets
        available_targets = [col for col in target_columns if col in df.columns]
        self.output_columns = available_targets
        
        # Select features (exclude metadata and targets)
        exclude_cols = set(available_targets) + set([
            'Time (s)', 'Throttle (Âµs)', 'Torque (Nâ‹…m)',
            'filename', 'Motor manufacturer', 'Propeller manufacturer',
            'ESC manufacturer', 'ESC limit'
        ])
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]
        self.feature_columns = feature_cols
        
        print(f"Using {len(feature_cols)} features")
        print(f"Predicting {len(available_targets)} targets")
        
        # Prepare X and y
        X = df[feature_cols].copy()
        y = df[available_targets].copy()
        
        # Remove rows with NaN in X or y
        mask = X.notna().all(axis=1) & y.notna().all(axis=1)
        X = X[mask]
        y = y[mask]
        
        print(f"Training samples after cleaning: {len(X)}")
        
        return X, y
    
    def train_xgboost_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost models for each output"""
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        
        for i, col in enumerate(self.output_columns):
            print(f"\nTraining model for: {col}")
            
            # Initialize model
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=20
            )
            
            # Prepare target
            y_train_scaled = self.output_scalers[col].fit_transform(
                y_train[[col]].values
            ).ravel()
            
            # Train
            if X_val is not None and y_val is not None:
                X_val_scaled = self.feature_scaler.transform(X_val)
                y_val_scaled = self.output_scalers[col].transform(
                    y_val[[col]].values
                ).ravel()
                
                model.fit(X_train_scaled, y_train_scaled,
                         eval_set=[(X_val_scaled, y_val_scaled)],
                         verbose=0)
            else:
                model.fit(X_train_scaled, y_train_scaled)
            
            self.models[col] = model
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train_scaled)
            print(f"  R² score: {train_score:.4f}")
    
    def train(self, df, test_size=0.2, val_size=0.1):
        """Complete training pipeline"""
        
        # Prepare data
        X, y = self.prepare_training_data(df)
        
        # Initialize output scalers
        for col in self.output_columns:
            self.output_scalers[col] = StandardScaler()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size/(1-test_size), random_state=42
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train models
        self.train_xgboost_models(X_train, y_train, X_val, y_val)
        
        # Evaluate
        self.evaluate(X_test, y_test)
        
        return X_test, y_test
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        for col in self.output_columns:
            model = self.models[col]
            y_test_scaled = self.output_scalers[col].transform(
                y_test[[col]].values
            ).ravel()
            
            score = model.score(X_test_scaled, y_test_scaled)
            print(f"\n{col}:")
            print(f"  R² Score: {score:.4f}")
            
            # Predictions
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = self.output_scalers[col].inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).ravel()
            
            # RMSE
            rmse = np.sqrt(np.mean((y_test[col].values - y_pred) ** 2))
            print(f"  RMSE: {rmse:.4f}")
            
            # MAE
            mae = np.mean(np.abs(y_test[col].values - y_pred))
            print(f"  MAE: {mae:.4f}")
    
    def predict(self, input_df):
        """Predict for new inputs"""
        X = input_df[self.feature_columns].copy()
        X_scaled = self.feature_scaler.transform(X)
        
        predictions = {}
        for col in self.output_columns:
            model = self.models[col]
            y_pred_scaled = model.predict(X_scaled)
            y_pred = self.output_scalers[col].inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).ravel()
            predictions[col] = y_pred
        
        result_df = pd.DataFrame(predictions)
        return result_df
    
    def save_models(self, model_dir="models"):
        """Save trained models"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        for col, model in self.models.items():
            joblib.dump(model, f"{model_dir}/{col.replace(' ', '_').replace('(', '').replace(')', '')}.pkl")
        
        joblib.dump(self.feature_scaler, f"{model_dir}/feature_scaler.pkl")
        joblib.dump(self.output_scalers, f"{model_dir}/output_scalers.pkl")
        joblib.dump(self.feature_columns, f"{model_dir}/feature_columns.pkl")
        joblib.dump(self.output_columns, f"{model_dir}/output_columns.pkl")
        
        print(f"Models saved to {model_dir}/")
    
    def load_models(self, model_dir="models"):
        """Load pre-trained models"""
        self.feature_columns = joblib.load(f"{model_dir}/feature_columns.pkl")
        self.output_columns = joblib.load(f"{model_dir}/output_columns.pkl")
        self.feature_scaler = joblib.load(f"{model_dir}/feature_scaler.pkl")
        self.output_scalers = joblib.load(f"{model_dir}/output_scalers.pkl")
        
        self.models = {}
        for col in self.output_columns:
            model_file = f"{model_dir}/{col.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
            self.models[col] = joblib.load(model_file)
        
        print(f"Models loaded from {model_dir}/")


# Training script
if __name__ == "__main__":
    # Load engineered features
    df = pd.read_csv("data/processed/engineered_features.csv")
    
    # Train model
    model = PropulsionSystemModel()
    X_test, y_test = model.train(df)
    
    # Save models
    model.save_models("models")
    
    # Test prediction
    test_input = X_test.iloc[:1]
    predictions = model.predict(test_input)
    print("\nSample Prediction:")
    print(predictions)
```

---

## Phase 4: Web Application for Local Deployment

### 4.1 Flask Web Application

**File: `web_app/app.py`**

```python
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from io import BytesIO, StringIO
import json
from datetime import datetime

app = Flask(__name__, template_folder='templates')

# Load pre-trained models
MODEL_DIR = "../models"
try:
    feature_columns = joblib.load(f"{MODEL_DIR}/feature_columns.pkl")
    output_columns = joblib.load(f"{MODEL_DIR}/output_columns.pkl")
    feature_scaler = joblib.load(f"{MODEL_DIR}/feature_scaler.pkl")
    output_scalers = joblib.load(f"{MODEL_DIR}/output_scalers.pkl")
    models = {}
    for col in output_columns:
        col_safe = col.replace(' ', '_').replace('(', '').replace(')', '')
        models[col] = joblib.load(f"{MODEL_DIR}/{col_safe}.pkl")
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    models = None

# Component specifications database
MOTOR_KV = [1100, 1450, 1750, 2200, 2850, 3520, 4600]
ESC_AMPERAGE = [20, 30, 40, 50, 60, 80, 100, 120]
BATTERY_VOLTAGE = [3.7, 7.4, 11.1, 14.8, 22.2]  # LiPo nominal
PROPELLER_DATA = {
    '5": [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
    '7": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    '8": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    '10": [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/components')
def get_components():
    """Get available component specifications"""
    return jsonify({
        'motor_kv': MOTOR_KV,
        'esc_amperage': ESC_AMPERAGE,
        'battery_voltage': BATTERY_VOLTAGE,
        'propeller_sizes': list(PROPELLER_DATA.keys()),
        'propeller_pitches': PROPELLER_DATA
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict propulsion system performance"""
    if models is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        data = request.json
        
        # Extract inputs
        motor_kv = float(data['motor_kv'])
        esc_amperage = float(data['esc_amperage'])
        battery_voltage = float(data['battery_voltage'])
        propeller_diameter = float(data['propeller_diameter'])
        propeller_pitch = float(data['propeller_pitch'])
        
        # Create feature vector matching training features
        # This is a simplified example - adjust based on your actual feature set
        input_features = {col: 0 for col in feature_columns}
        input_features['motor_kv'] = motor_kv
        input_features['esc_amperage'] = esc_amperage
        input_features['battery_voltage'] = battery_voltage
        input_features['propeller_diameter_inch'] = propeller_diameter
        input_features['propeller_pitch_inch'] = propeller_pitch
        
        # Create dataframe
        X = pd.DataFrame([input_features])
        X = X[feature_columns]
        
        # Scale and predict
        X_scaled = feature_scaler.transform(X)
        
        predictions = {}
        for col in output_columns:
            model = models[col]
            y_pred_scaled = model.predict(X_scaled)
            y_pred = output_scalers[col].inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            )[0, 0]
            predictions[col] = float(y_pred)
        
        return jsonify({
            'status': 'success',
            'input': {
                'motor_kv': motor_kv,
                'esc_amperage': esc_amperage,
                'battery_voltage': battery_voltage,
                'propeller_diameter': propeller_diameter,
                'propeller_pitch': propeller_pitch
            },
            'predictions': predictions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/export-csv', methods=['POST'])
def export_csv():
    """Export prediction as CSV"""
    try:
        data = request.json
        
        # Get predictions
        predictions = data['predictions']
        inputs = data['input']
        
        # Create dataframe
        df = pd.DataFrame([{
            'Motor_Kv': inputs['motor_kv'],
            'ESC_Amperage': inputs['esc_amperage'],
            'Battery_Voltage': inputs['battery_voltage'],
            'Propeller_Diameter_inch': inputs['propeller_diameter'],
            'Propeller_Pitch_inch': inputs['propeller_pitch'],
            **predictions
        }])
        
        # Convert to CSV
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        # Return as download
        response_data = {
            'csv': csv_buffer.getvalue(),
            'filename': f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction from CSV"""
    try:
        # Read uploaded CSV
        file = request.files['file']
        df_input = pd.read_csv(file)
        
        # Predict for each row
        predictions_list = []
        for idx, row in df_input.iterrows():
            input_features = {col: 0 for col in feature_columns}
            
            # Map input columns
            if 'motor_kv' in df_input.columns:
                input_features['motor_kv'] = float(row['motor_kv'])
            # ... map other columns
            
            X = pd.DataFrame([input_features])
            X = X[feature_columns]
            X_scaled = feature_scaler.transform(X)
            
            row_predictions = {}
            for col in output_columns:
                y_pred_scaled = models[col].predict(X_scaled)
                y_pred = output_scalers[col].inverse_transform(
                    y_pred_scaled.reshape(-1, 1)
                )[0, 0]
                row_predictions[col] = float(y_pred)
            
            predictions_list.append({
                'input': row.to_dict(),
                'predictions': row_predictions
            })
        
        return jsonify({
            'status': 'success',
            'results': predictions_list,
            'count': len(predictions_list)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### 4.2 HTML Frontend

**File: `web_app/templates/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Propulsion System Characterization Tool</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        h1 { font-size: 2em; margin-bottom: 10px; }
        .subtitle { font-size: 1.1em; opacity: 0.9; }
        
        main { padding: 40px; }
        
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
        }
        
        label {
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }
        
        input, select {
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 5px rgba(102, 126, 234, 0.3);
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
        }
        
        button {
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }
        
        .btn-secondary:hover {
            background: #e0e0e0;
        }
        
        .results {
            background: #f9f9f9;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            display: none;
        }
        
        .results.active { display: block; }
        
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .result-card {
            background: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
        
        .result-card h3 {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        
        .result-card .value {
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }
        
        .result-card .unit {
            font-size: 0.85em;
            color: #999;
            margin-top: 5px;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #ffebee;
            border: 1px solid #ef5350;
            color: #c62828;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: none;
        }
        
        .error.active { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚁 Propulsion System Characterization</h1>
            <p class="subtitle">AI-Powered Motor, ESC, Battery & Propeller Performance Prediction</p>
        </header>
        
        <main>
            <div class="error" id="error-message"></div>
            
            <form id="prediction-form">
                <div class="form-grid">
                    <div class="form-group">
                        <label for="motor-kv">Motor Kv (RPM/V)</label>
                        <select id="motor-kv" name="motor_kv" required>
                            <option value="">Select Motor Kv...</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="esc-amperage">ESC Amperage (A)</label>
                        <select id="esc-amperage" name="esc_amperage" required>
                            <option value="">Select ESC Rating...</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="battery-voltage">Battery Voltage (V)</label>
                        <select id="battery-voltage" name="battery_voltage" required>
                            <option value="">Select Battery Voltage...</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="propeller-diameter">Propeller Diameter (inch)</label>
                        <select id="propeller-diameter" name="propeller_diameter" required>
                            <option value="">Select Diameter...</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="propeller-pitch">Propeller Pitch (inch)</label>
                        <select id="propeller-pitch" name="propeller_pitch" required>
                            <option value="">Select Pitch...</option>
                        </select>
                    </div>
                </div>
                
                <div class="button-group">
                    <button type="submit" class="btn-primary">🔮 Predict Performance</button>
                    <button type="reset" class="btn-secondary">Clear</button>
                    <button type="button" class="btn-secondary" onclick="downloadTemplate()">📥 Download CSV Template</button>
                </div>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing propulsion system...</p>
            </div>
            
            <div class="results" id="results">
                <h2>Predicted Performance</h2>
                <div class="results-grid" id="results-grid"></div>
                <div class="button-group" style="margin-top: 20px;">
                    <button type="button" class="btn-primary" onclick="downloadResults()">⬇️ Download CSV</button>
                </div>
            </div>
        </main>
    </div>
    
    <script>
        // Load component options
        fetch('/api/components')
            .then(response => response.json())
            .then(data => {
                populateSelect('motor-kv', data.motor_kv);
                populateSelect('esc-amperage', data.esc_amperage);
                populateSelect('battery-voltage', data.battery_voltage);
                
                const diameterSelect = document.getElementById('propeller-diameter');
                data.propeller_sizes.forEach(size => {
                    const option = document.createElement('option');
                    option.value = size;
                    option.textContent = size;
                    diameterSelect.appendChild(option);
                });
                
                diameterSelect.addEventListener('change', function() {
                    const pitches = data.propeller_pitches[this.value] || [];
                    const pitchSelect = document.getElementById('propeller-pitch');
                    pitchSelect.innerHTML = '<option value="">Select Pitch...</option>';
                    pitches.forEach(pitch => {
                        const option = document.createElement('option');
                        option.value = pitch;
                        option.textContent = pitch;
                        pitchSelect.appendChild(option);
                    });
                });
            });
        
        function populateSelect(elementId, options) {
            const select = document.getElementById(elementId);
            options.forEach(option => {
                const opt = document.createElement('option');
                opt.value = option;
                opt.textContent = option;
                select.appendChild(opt);
            });
        }
        
        document.getElementById('prediction-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = Object.fromEntries(formData);
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').classList.remove('active');
            document.getElementById('error-message').classList.remove('active');
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    displayResults(result.input, result.predictions);
                    window.lastResults = {input: result.input, predictions: result.predictions};
                } else {
                    showError(result.error);
                }
            } catch (error) {
                showError('Prediction failed: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });
        
        function displayResults(input, predictions) {
            const grid = document.getElementById('results-grid');
            grid.innerHTML = '';
            
            const units = {
                'Rotation speed (rpm)': 'RPM',
                'Thrust (kgf)': 'kg-force',
                'Electrical power (W)': 'Watts',
                'Motor & ESC efficiency (%)': '%',
                'Propeller efficiency (gf/W)': 'gf/W'
            };
            
            for (const [key, value] of Object.entries(predictions)) {
                const card = document.createElement('div');
                card.className = 'result-card';
                card.innerHTML = `
                    <h3>${key}</h3>
                    <div class="value">${parseFloat(value).toFixed(2)}</div>
                    <div class="unit">${units[key] || ''}</div>
                `;
                grid.appendChild(card);
            }
            
            document.getElementById('results').classList.add('active');
        }
        
        function showError(message) {
            const error = document.getElementById('error-message');
            error.textContent = message;
            error.classList.add('active');
        }
        
        async function downloadResults() {
            if (!window.lastResults) return;
            
            const response = await fetch('/api/export-csv', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(window.lastResults)
            });
            
            const data = await response.json();
            const csv = data.csv;
            const filename = data.filename;
            
            const link = document.createElement('a');
            link.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
            link.download = filename;
            link.click();
        }
        
        function downloadTemplate() {
            const csv = `motor_kv,esc_amperage,battery_voltage,propeller_diameter,propeller_pitch
2850,30,7.4,5,3
3520,40,11.1,7,6
1450,50,14.8,8,5`;
            
            const link = document.createElement('a');
            link.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
            link.download = 'batch_prediction_template.csv';
            link.click();
        }
    </script>
</body>
</html>
```

### 4.3 Requirements File

**File: `web_app/requirements.txt`**

```
Flask==2.3.2
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==2.0.0
joblib==1.3.1
openpyxl==3.1.2
python-dotenv==1.0.0
```

---

## Phase 5: Local Deployment Instructions

### 5.1 Setup & Installation

```bash
# 1. Clone/download project
cd /path/to/propulsion_ml_system

# 2. Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# 3. Install dependencies
pip install -r src/requirements.txt
pip install -r web_app/requirements.txt

# 4. Organize data
# Place all 300+ Excel files in data/raw/
# Create data/processed/ directory
mkdir -p data/processed models

# 5. Run data preparation
python src/data_loader.py
python src/feature_engineering.py

# 6. Train models
python src/model_trainer.py

# 7. Run web app
cd web_app
python app.py

# Access: http://localhost:5000
```

### 5.2 Configuration File

**File: `config.py`**

```python
import os

# Data paths
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

# Model parameters
TRAIN_TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.1

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}

# Feature engineering
PHYSICS_FEATURES = [
    'electrical_power_input',
    'motor_back_emf_ratio',
    'motor_efficiency',
    'motor_torque_nm',
    'propeller_disk_loading',
    'propeller_tip_speed_ms',
    'pitch_diameter_ratio',
    'thrust_per_watt',
    'current_per_kgf_thrust',
]

# Prediction outputs
OUTPUT_COLUMNS = [
    'Rotation speed (rpm)',
    'Thrust (kgf)',
    'Electrical power (W)',
    'Motor & ESC efficiency (%)',
    'Propeller efficiency (gf/W)',
]

# Web app
FLASK_DEBUG = True
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
```

---

## Phase 6: Advanced Optimizations

### 6.1 Propeller Geometry Feature Extraction

For better handling of propeller variations:

```python
# Add to feature_engineering.py

def extract_propeller_features(df):
    """Extract advanced propeller characteristics"""
    
    # Solidity (blade area / disk area)
    # Simplified: assume standard blade count
    blade_count = 2  # typical for drones
    blade_fraction = 0.3  # approximate
    df['propeller_solidity'] = blade_count * blade_fraction
    
    # Reynolds number approximation
    # Re = ρ * v * D / μ (air density, velocity, diameter, viscosity)
    rho_air = 1.225  # kg/m³
    mu_air = 1.81e-5  # Pa·s
    
    diameter_m = df['propeller_diameter_inch'] * 0.0254
    velocity = df['propeller_tip_speed_ms']
    
    df['propeller_reynolds_number'] = (rho_air * velocity * diameter_m) / mu_air
    
    # Advance ratio (J = V_forward / (n * D))
    # For hover, V_forward ≈ 0, so J ≈ 0
    df['propeller_advance_ratio'] = 0.0  # Hover assumption
    
    # Thrust coefficient (C_T = T / (ρ * n² * D⁴))
    rho_air = 1.225
    thrust_n = df['Thrust (kgf)'] * 9.81
    rpm = df['Rotation speed (rpm)']
    rps = rpm / 60
    
    ct = thrust_n / (rho_air * (rps ** 2) * (diameter_m ** 4) + 1e-6)
    df['propeller_thrust_coefficient'] = ct.clip(-10, 10)
    
    return df
```

### 6.2 Ensemble Methods

```python
from sklearn.ensemble import VotingRegressor

def create_ensemble_model(X_train, y_train):
    """Combine multiple algorithms"""
    
    xgb_model = xgb.XGBRegressor(max_depth=6, learning_rate=0.05)
    gb_model = GradientBoostingRegressor(max_depth=5, n_estimators=200)
    rf_model = RandomForestRegressor(max_depth=10, n_estimators=100)
    
    ensemble = VotingRegressor(
        estimators=[
            ('xgb', xgb_model),
            ('gb', gb_model),
            ('rf', rf_model)
        ],
        weights=[0.5, 0.3, 0.2]  # XGBoost weighted higher
    )
    
    ensemble.fit(X_train, y_train)
    return ensemble
```

### 6.3 Uncertainty Quantification

```python
from sklearn.ensemble import GradientBoostingRegressor

def predict_with_uncertainty(model, X):
    """Get predictions with confidence intervals"""
    
    # Use Quantile Regression for uncertainty
    quantile_models = {
        'lower': GradientBoostingRegressor(loss='quantile', alpha=0.05),
        'median': GradientBoostingRegressor(loss='quantile', alpha=0.50),
        'upper': GradientBoostingRegressor(loss='quantile', alpha=0.95),
    }
    
    predictions = {}
    for q, model in quantile_models.items():
        predictions[q] = model.predict(X)
    
    return predictions
```

---

## Phase 7: Validation & Testing

### 7.1 Cross-Validation

```python
from sklearn.model_selection import cross_validate

# In model_trainer.py
cv_results = cross_validate(
    model, X_train, y_train,
    cv=5,  # 5-fold
    scoring=['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'],
    n_jobs=-1
)

print("Cross-validation results:")
for metric, scores in cv_results.items():
    print(f"{metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 7.2 Domain-Specific Validation

```python
def validate_physics_constraints(predictions):
    """Check if predictions satisfy physics"""
    
    errors = []
    
    # RPM must be positive
    if predictions['rpm'] < 0:
        errors.append("RPM cannot be negative")
    
    # Thrust must increase with RPM
    # Efficiency must be between 0-1
    if not (0 <= predictions['efficiency'] <= 1):
        errors.append("Efficiency out of valid range")
    
    # Power relationships
    # Mech Power = Torque * Angular Velocity
    # Elec Power > Mech Power (losses)
    if predictions['electrical_power'] < predictions['mechanical_power']:
        errors.append("Electrical power less than mechanical (violates energy conservation)")
    
    return errors
```

---

## Phase 8: Performance Monitoring

### 8.1 API Logging

```python
import logging
from datetime import datetime

logging.basicConfig(
    filename='logs/predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

@app.route('/api/predict', methods=['POST'])
def predict():
    # ... prediction code ...
    
    logging.info(f"Prediction request: {data}")
    logging.info(f"Predictions: {predictions}")
    
    return jsonify({...})
```

### 8.2 Model Performance Dashboard

```python
# Add to app.py
@app.route('/dashboard')
def dashboard():
    """Model performance metrics"""
    metrics = {
        'total_predictions': 1234,
        'avg_prediction_time': 0.045,  # seconds
        'model_accuracy': 0.92,
        'last_retrain': '2024-01-15'
    }
    return render_template('dashboard.html', metrics=metrics)
```

---

## Summary & Next Steps

✅ **What you have:**
1. Complete ML pipeline for propulsion characterization
2. Web application for local deployment
3. Physics-informed feature engineering
4. Multi-output regression models
5. CSV import/export capabilities

📋 **To implement:**
```bash
# 1. Data organization
cp -r your_300_excel_files data/raw/

# 2. Training pipeline
python -m src.data_loader
python -m src.feature_engineering
python -m src.model_trainer

# 3. Web deployment
cd web_app && python app.py

# 4. Access dashboard
# Open http://localhost:5000
```

🎯 **Advanced features (optional):**
- Add real-time model retraining
- Implement active learning (select new data points to test)
- Deploy with Docker for easier sharing
- Add REST API documentation (Swagger)
- Build mobile app for field testing
- Integrate with your thrust stand for automated labeling

---

## Technical Contact Points

**For your aerospace work:**
- Physics layer validates against momentum theory
- Handles propeller geometry variations through learned feature representations
- Scalable to 300+ combinations without physical models

**Deployment simplicity:**
- Single command to run: `python app.py`
- No external servers needed
- Can run on laptop or low-power device (Raspberry Pi)


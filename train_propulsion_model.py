#!/usr/bin/env python3
"""
Propulsion System ML Model Training Framework
Train a surrogate model on 300+ experimental CSV files to predict performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from typing import Dict, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: DATA PREPROCESSING
# ============================================================================

class ThrustStandDataProcessor:
    """
    Load and standardize 300+ experimental CSV files from thrust stand testing.
    
    Expected CSV format:
    - Columns: throttle, throttle_percent, rpm, thrust, torque, voltage, 
               current, electrical_power, motor_mfg, motor_kv, prop_diameter, 
               prop_pitch, prop_mfg, esc_mfg, esc_limit
    """
    
    def __init__(self, csv_folder_path: str):
        self.csv_folder = Path(csv_folder_path)
        self.all_data = []
        self.metadata = {}
        
    def extract_metadata_from_filename(self, filename: str) -> Dict:
        """
        Extract motor KV, ESC A, Battery V, Prop D, Prop P from filename.
        
        Expected formats:
        - "T-Motor_2850KV_30A_11.1V_6x4_APC.csv"
        - "MWD_900KV_40A_22.2V_12x4_MAS.csv"
        """
        metadata = {}
        try:
            # Try pattern: Mfg_KVKV_AA_VV_DxP_PropMfg.csv
            parts = filename.replace('.csv', '').split('_')
            
            for part in parts:
                if 'kv' in part.lower():
                    metadata['motor_kv'] = float(part.replace('KV', '').replace('kv', ''))
                elif 'a' in part.lower() and 'kv' not in part.lower():
                    metadata['esc_limit_a'] = float(part.replace('A', '').replace('a', ''))
                elif 'v' in part.lower() and 'kv' not in part.lower():
                    metadata['battery_voltage'] = float(part.replace('V', '').replace('v', ''))
                elif 'x' in part.lower():
                    props = part.replace('x', 'X').split('X')
                    if len(props) == 2:
                        metadata['prop_diameter'] = float(props[0])
                        metadata['prop_pitch'] = float(props[1])
            
            return metadata if len(metadata) >= 4 else {}
        except:
            return {}
    
    def load_all_csvs(self) -> pd.DataFrame:
        """Load and merge all CSV files from folder"""
        print(f"Loading CSV files from: {self.csv_folder}")
        
        csv_files = list(self.csv_folder.glob('*.csv'))
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Extract metadata
                metadata = self.extract_metadata_from_filename(csv_file.name)
                
                # If metadata extraction failed, try reading from CSV columns
                if not metadata:
                    metadata = {
                        'motor_kv': df['motor_kv'].iloc[0] if 'motor_kv' in df.columns else None,
                        'esc_limit_a': df['esc_limit'].iloc[0] if 'esc_limit' in df.columns else None,
                        'prop_diameter': df['prop_diameter'].iloc[0] if 'prop_diameter' in df.columns else None,
                        'prop_pitch': df['prop_pitch'].iloc[0] if 'prop_pitch' in df.columns else None,
                        'battery_voltage': df.get('battery_voltage', {}).iloc[0] if 'battery_voltage' in df.columns else 11.1
                    }
                
                # Add metadata to dataframe
                for key, value in metadata.items():
                    df[key] = value
                
                self.all_data.append(df)
                print(f"  ✓ Loaded: {csv_file.name} | KV={metadata.get('motor_kv')} ESC={metadata.get('esc_limit_a')}A")
                
            except Exception as e:
                print(f"  ✗ Error loading {csv_file.name}: {e}")
        
        if self.all_data:
            combined_df = pd.concat(self.all_data, ignore_index=True)
            print(f"\nCombined dataset: {combined_df.shape[0]} rows × {combined_df.shape[1]} columns")
            return combined_df
        else:
            print("No CSV files loaded!")
            return pd.DataFrame()
    
    def standardize_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert all measurements to SI units"""
        
        # Propeller dimensions: inches → meters
        if 'prop_diameter' in df.columns:
            df['prop_diameter_m'] = df['prop_diameter'] * 0.0254
        if 'prop_pitch' in df.columns:
            df['prop_pitch_m'] = df['prop_pitch'] * 0.0254
        
        # Thrust: grams → kg
        if 'thrust' in df.columns and df['thrust'].max() > 50:
            df['thrust_kg'] = df['thrust'] / 1000
        elif 'thrust' in df.columns:
            df['thrust_kg'] = df['thrust']
        
        # Power: ensure Watts
        if 'electrical_power' in df.columns:
            df['power_w'] = df['electrical_power']
        elif 'power' in df.columns:
            df['power_w'] = df['power']
        
        # Voltage: ensure Volts (already SI)
        if 'voltage' in df.columns:
            df['voltage_v'] = df['voltage']
        
        # Current: ensure Amps (already SI)
        if 'current' in df.columns:
            df['current_a'] = df['current']
        
        # Torque: ensure N⋅m
        if 'torque' in df.columns:
            df['torque_nm'] = df['torque']
        
        # Throttle percentage
        if 'throttle_percent' in df.columns:
            df['throttle_percentage'] = df['throttle_percent']
        elif 'throttle' in df.columns:
            df['throttle_percentage'] = (df['throttle'] - 1000) / 10
        
        return df

# ============================================================================
# PART 2: FEATURE ENGINEERING
# ============================================================================

class AerodynamicFeatureEngineer:
    """Create physics-based features for propulsion system modeling"""
    
    RHO = 1.225  # Air density at sea level (kg/m³)
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def engineer_all_features(self) -> pd.DataFrame:
        """Create comprehensive feature set"""
        
        print("\n[FEATURE ENGINEERING]")
        
        # Electrical features
        self.df['max_rpm'] = self.df['motor_kv'] * self.df['battery_voltage']
        self.df['rpm_percentage'] = (self.df['rpm'] / self.df['max_rpm']).clip(0, 1)
        self.df['max_power_w'] = self.df['battery_voltage'] * self.df['esc_limit_a']
        
        # Propeller geometry features
        self.df['prop_area_m2'] = np.pi * (self.df['prop_diameter_m'] / 2) ** 2
        self.df['pitch_diameter_ratio'] = self.df['prop_pitch_m'] / self.df['prop_diameter_m']
        self.df['aspect_ratio'] = (self.df['prop_diameter_m'] ** 2) / (self.df['prop_area_m2'] + 1e-6)
        
        # Aerodynamic coefficients
        self.df['thrust_coefficient'] = np.clip(
            (self.df['thrust_kg'] * 9.81) / 
            (self.RHO * self.df['prop_area_m2'] * (self.df['rpm'] / 60) ** 2 + 1e-6),
            -1, 1
        )
        
        self.df['power_coefficient'] = np.clip(
            self.df['power_w'] / 
            (self.RHO * self.df['prop_area_m2'] * (self.df['rpm'] / 60) ** 3 + 1e-6),
            0, 2
        )
        
        # Mechanical features
        self.df['tip_speed_ms'] = np.pi * self.df['prop_diameter_m'] * (self.df['rpm'] / 60)
        self.df['torque_from_power'] = (self.df['power_w'] * 60) / (2 * np.pi * self.df['rpm'] + 1e-6)
        
        # Electrical efficiency
        self.df['input_power_w'] = self.df['voltage_v'] * self.df['current_a']
        self.df['motor_esc_efficiency'] = np.clip(
            self.df['power_w'] / (self.df['input_power_w'] + 1e-6),
            0, 1
        )
        
        # Interaction terms
        self.df['kv_voltage_product'] = self.df['motor_kv'] * self.df['battery_voltage']
        self.df['diameter_pitch_product'] = self.df['prop_diameter'] * self.df['prop_pitch']
        
        print(f"  ✓ Created {len(self.df.columns)} total features")
        
        return self.df

# ============================================================================
# PART 3: MODEL TRAINING
# ============================================================================

class PropulsionSurrogateModel:
    """Train multi-output XGBoost model for propulsion systems"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.models = {}
        self.scalers = {}
        self.output_names = ['rpm', 'thrust_kg', 'power_w', 'efficiency', 
                           'current_a', 'voltage_v', 'torque_nm']
        self.input_features = ['motor_kv', 'esc_limit_a', 'battery_voltage',
                             'prop_diameter_m', 'prop_pitch_m', 'throttle_percentage']
        
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Prepare and normalize training data"""
        
        print("\n[DATA PREPARATION]")
        
        # Select features and outputs
        X = self.df[self.input_features].copy()
        y = self.df[self.output_names].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # Remove outliers (3-sigma rule)
        for col in y.columns:
            mean = y[col].mean()
            std = y[col].std()
            mask = (y[col] > mean - 3*std) & (y[col] < mean + 3*std)
            X = X[mask]
            y = y[mask]
        
        print(f"  ✓ After outlier removal: {X.shape[0]} samples")
        
        # Normalize features
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        
        # Normalize targets
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y)
        
        print(f"  ✓ Features normalized (shape: {X_scaled.shape})")
        print(f"  ✓ Targets normalized (shape: {y_scaled.shape})")
        
        self.scalers = {'X': scaler_X, 'y': scaler_y}
        
        return X_scaled, y_scaled, {'X': X, 'y': y}
    
    def train(self):
        """Train separate XGBoost model for each output"""
        
        print("\n[MODEL TRAINING]")
        
        X_scaled, y_scaled, _ = self.prepare_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        
        # Train models
        for idx, output_name in enumerate(self.output_names):
            print(f"\n  Training {output_name}...")
            
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(
                X_train, y_train[:, idx],
                eval_set=[(X_test, y_test[:, idx])],
                verbose=False
            )
            
            # Evaluate
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test[:, idx], y_pred))
            mae = mean_absolute_error(y_test[:, idx], y_pred)
            r2 = r2_score(y_test[:, idx], y_pred)
            
            self.models[output_name] = model
            
            print(f"    RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
        
        print("\n  ✓ All models trained successfully!")
        return self
    
    def save(self, filepath: str = 'propulsion_surrogate_model.pkl'):
        """Save trained model and scalers"""
        
        data = {
            'models': self.models,
            'scalers': self.scalers,
            'input_features': self.input_features,
            'output_names': self.output_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n✓ Model saved to: {filepath}")

# ============================================================================
# PART 4: PREDICTION INTERFACE
# ============================================================================

class PredictionEngine:
    """Generate predictions for new motor-ESC-battery-propeller combinations"""
    
    def __init__(self, model_filepath: str):
        with open(model_filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.models = data['models']
        self.scalers = data['scalers']
        self.input_features = data['input_features']
        self.output_names = data['output_names']
    
    def predict_throttle_curve(self, motor_kv: float, esc_limit_a: float,
                             battery_voltage: float, prop_diameter_in: float,
                             prop_pitch_in: float, num_points: int = 11) -> pd.DataFrame:
        """
        Generate performance curve across throttle range.
        
        Returns DataFrame matching your experimental CSV format.
        """
        
        throttle_levels = np.linspace(1000, 2000, num_points)
        throttle_percentages = np.linspace(0, 100, num_points)
        
        predictions = []
        
        for throttle, throttle_pct in zip(throttle_levels, throttle_percentages):
            # Convert units
            prop_diameter_m = prop_diameter_in * 0.0254
            prop_pitch_m = prop_pitch_in * 0.0254
            
            # Create feature vector
            X_new = np.array([[
                motor_kv,
                esc_limit_a,
                battery_voltage,
                prop_diameter_m,
                prop_pitch_m,
                throttle_pct
            ]])
            
            # Normalize
            X_scaled = self.scalers['X'].transform(X_new)
            
            # Predict each output
            outputs = {}
            for output_name in self.output_names:
                model = self.models[output_name]
                y_pred = model.predict(X_scaled)[0]
                outputs[output_name] = y_pred
            
            row = {
                'throttle': int(throttle),
                'throttle_percent': f"{throttle_pct:.1f}",
                'rpm': int(outputs['rpm']),
                'thrust_kg': outputs['thrust_kg'],
                'torque_nm': outputs['torque_nm'],
                'voltage_v': outputs['voltage_v'],
                'current_a': outputs['current_a'],
                'electrical_power': outputs['power_w'],
                'motor_kv': motor_kv,
                'esc_limit': esc_limit_a,
                'prop_diameter': prop_diameter_in,
                'prop_pitch': prop_pitch_in
            }
            
            predictions.append(row)
        
        return pd.DataFrame(predictions)

# ============================================================================
# QUICK START EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("PROPULSION SYSTEM ML MODEL TRAINING")
    print("="*70)
    
    # Step 1: Load and process data
    print("\n[STEP 1: LOAD DATA]")
    processor = ThrustStandDataProcessor('./thrust_test_data')  # Your CSV folder
    df = processor.load_all_csvs()
    
    if df.empty:
        print("No data loaded. Please ensure CSV files are in './thrust_test_data' folder")
        exit(1)
    
    # Step 2: Standardize units
    print("\n[STEP 2: STANDARDIZE UNITS]")
    df = processor.standardize_units(df)
    print("  ✓ All units converted to SI")
    
    # Step 3: Engineer features
    print("\n[STEP 3: ENGINEER FEATURES]")
    engineer = AerodynamicFeatureEngineer(df)
    df_features = engineer.engineer_all_features()
    
    # Step 4: Train model
    print("\n[STEP 4: TRAIN MODEL]")
    model = PropulsionSurrogateModel(df_features)
    model.train()
    
    # Step 5: Save model
    model.save('propulsion_model.pkl')
    
    # Step 6: Test predictions
    print("\n" + "="*70)
    print("GENERATING SAMPLE PREDICTIONS")
    print("="*70)
    
    engine = PredictionEngine('propulsion_model.pkl')
    
    # Example: T-Motor 2850KV with 30A ESC, 11.1V battery, 6x4 propeller
    result_df = engine.predict_throttle_curve(
        motor_kv=2850,
        esc_limit_a=30,
        battery_voltage=11.1,
        prop_diameter_in=6,
        prop_pitch_in=4,
        num_points=11
    )
    
    print("\nSample Predictions (2850KV, 30A, 11.1V, 6x4 prop):")
    print(result_df[['throttle', 'rpm', 'thrust_kg', 'power_w']].to_string(index=False))
    
    # Save as CSV
    result_df.to_csv('predicted_performance.csv', index=False)
    print("\n✓ Predictions saved to: predicted_performance.csv")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)

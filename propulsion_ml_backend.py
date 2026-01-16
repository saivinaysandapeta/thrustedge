
"""
Propulsion System ML Model Backend
=================================
Train ML models on experimental thrust stand data (300+ CSVs)
and generate predictions for new motor/prop/ESC/battery combinations
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import joblib
from datetime import datetime

class PropulsionModelPipeline:
    """
    Complete ML pipeline for propulsion system performance prediction
    """

    def __init__(self, model_dir='./models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        self.models = {}
        self.scaler = None
        self.feature_names = [
            'motor_kv', 'esc_limit_a', 'battery_voltage_v', 
            'prop_diameter_in', 'prop_pitch_in', 'throttle_pct'
        ]

    def load_experimental_data(self, csv_directory):
        """
        Load all CSV files from experimental thrust stand runs

        Expected CSV structure:
        Time,Throttle (µs),Rotation speed (rpm),Thrust (kgf),Torque (N⋅m),
        Voltage (V),Current (A),Electrical power (W),Mechanical power (W),...
        """
        csv_files = list(Path(csv_directory).glob('*.csv'))
        print(f"Found {len(csv_files)} CSV files")

        all_data = []

        for csv_file in csv_files:
            try:
                # Parse filename to extract configuration
                # Format: motor_KV2850_esc_30A_battery_3S_prop_6x4_APC.csv
                config = self.parse_filename(csv_file.stem)

                # Load CSV
                df = pd.read_csv(csv_file)

                # Standardize column names
                df = self.standardize_columns(df)

                # Add configuration parameters
                for key, value in config.items():
                    df[key] = value

                # Feature engineering
                df = self.engineer_features(df)

                all_data.append(df)
                print(f"  ✓ Loaded: {csv_file.name}")

            except Exception as e:
                print(f"  ✗ Error loading {csv_file.name}: {e}")

        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"\nTotal samples: {len(combined_data)}")
            return combined_data
        return None

    def parse_filename(self, filename):
        """
        Parse configuration from standardized filename
        Example: motor_KV2850_esc_30A_battery_3S_prop_6x4_APC
        """
        parts = filename.split('_')
        config = {
            'motor_kv': 2850,
            'esc_limit_a': 30,
            'battery_voltage_v': 11.1,
            'prop_diameter_in': 6,
            'prop_pitch_in': 4,
            'prop_mfg': 'Unknown'
        }

        # Parse available fields
        try:
            for i, part in enumerate(parts):
                if 'KV' in part.upper() and i+1 < len(parts):
                    config['motor_kv'] = float(parts[i+1].replace('KV', ''))
                elif 'ESC' in part.upper() and i+1 < len(parts):
                    config['esc_limit_a'] = float(parts[i+1].replace('A', ''))
                elif 'BATTERY' in part.upper() and i+1 < len(parts):
                    # Handle "3S" format
                    cells = int(parts[i+1].replace('S', ''))
                    config['battery_voltage_v'] = cells * 3.7
                elif 'PROP' in part.upper() and i+2 < len(parts):
                    prop_size = parts[i+1]  # e.g., "6x4"
                    if 'x' in prop_size:
                        dia, pitch = prop_size.split('x')
                        config['prop_diameter_in'] = float(dia)
                        config['prop_pitch_in'] = float(pitch)
                    config['prop_mfg'] = parts[i+2] if i+2 < len(parts) else 'Unknown'
        except:
            pass

        return config

    def standardize_columns(self, df):
        """
        Standardize column names from various CSV formats
        """
        column_mapping = {
            'Throttle (µs)': 'throttle_us',
            'Throttle (Âµs)': 'throttle_us',
            'Rotation speed (rpm)': 'rpm',
            'Thrust (kgf)': 'thrust_kgf',
            'Torque (Nâ‹…m)': 'torque_nm',
            'Torque (N⋅m)': 'torque_nm',
            'Voltage (V)': 'voltage_v',
            'Current (A)': 'current_a',
            'Electrical power (W)': 'electrical_power_w',
            'Mechanical power (W)': 'mechanical_power_w',
        }

        df.rename(columns=column_mapping, inplace=True)
        return df

    def engineer_features(self, df):
        """
        Create additional features from raw data
        """
        # Throttle percentage
        if 'throttle_us' in df.columns:
            df['throttle_pct'] = (df['throttle_us'] - 1000) / 10

        # Propeller aerodynamic features
        if 'prop_diameter_in' in df.columns:
            prop_diameter_m = df['prop_diameter_in'] * 0.0254
            df['prop_disk_area_m2'] = np.pi * (prop_diameter_m / 2) ** 2

            if 'thrust_kgf' in df.columns:
                df['thrust_per_disk_area'] = df['thrust_kgf'] / (df['prop_disk_area_m2'] + 0.001)

        # Motor constants
        if 'motor_kv' in df.columns:
            df['motor_torque_constant'] = 1 / (df['motor_kv'] * 0.00955)  # Kt = 1/(10*KV)

        # Efficiency features
        if 'electrical_power_w' in df.columns and 'thrust_kgf' in df.columns:
            df['thrust_per_watt'] = (df['thrust_kgf'] * 1000) / (df['electrical_power_w'] + 0.1)

        return df

    def train(self, data):
        """
        Train ensemble of models on experimental data
        """
        print("\n" + "="*80)
        print("TRAINING ML MODELS")
        print("="*80)

        # Prepare features
        X = data[self.feature_names].copy()

        # Fill any missing values
        X = X.fillna(X.mean())

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, train_idx, test_idx = train_test_split(
            X_scaled, np.arange(len(X)), test_size=0.15, random_state=42
        )

        # Prepare targets
        targets = {
            'rpm': data.loc[train_idx, 'rpm'].values if 'rpm' in data.columns else None,
            'thrust_kgf': data.loc[train_idx, 'thrust_kgf'].values if 'thrust_kgf' in data.columns else None,
            'torque_nm': data.loc[train_idx, 'torque_nm'].values if 'torque_nm' in data.columns else None,
            'current_a': data.loc[train_idx, 'current_a'].values if 'current_a' in data.columns else None,
            'electrical_power_w': data.loc[train_idx, 'electrical_power_w'].values if 'electrical_power_w' in data.columns else None,
            'voltage_v': data.loc[train_idx, 'voltage_v'].values if 'voltage_v' in data.columns else None,
        }

        # Train models
        model_configs = {
            'rpm': GradientBoostingRegressor(n_estimators=200, max_depth=7, learning_rate=0.1, random_state=42),
            'thrust_kgf': GradientBoostingRegressor(n_estimators=200, max_depth=7, learning_rate=0.1, random_state=42),
            'torque_nm': GradientBoostingRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42),
            'current_a': GradientBoostingRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42),
            'electrical_power_w': GradientBoostingRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42),
            'voltage_v': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        }

        for target_name, target_data in targets.items():
            if target_data is None:
                continue

            model = model_configs[target_name]

            # Train
            model.fit(X_train, target_data)

            # Evaluate
            y_test = data.loc[test_idx, target_name].values
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred + 1e-6)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, target_data, cv=5, scoring='r2')

            self.models[target_name] = model

            print(f"\n{target_name.upper()}")
            print(f"  R² Score: {r2:.4f}")
            print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  MAPE: {mape:.4f}%")

            # Feature importance
            importances = model.feature_importances_
            top_features = np.argsort(importances)[-3:][::-1]
            print(f"  Top features: {', '.join([self.feature_names[i] for i in top_features])}")

        # Save scaler
        joblib.dump(self.scaler, self.model_dir / 'scaler.pkl')
        print(f"\nModels saved to {self.model_dir}")

    def predict(self, motor_kv, esc_limit_a, battery_voltage_v, 
                prop_diameter_in, prop_pitch_in, throttle_pct):
        """
        Generate single prediction for given parameters
        """
        if self.scaler is None or not self.models:
            raise ValueError("Models not trained. Call train() first.")

        # Prepare input
        X = np.array([[motor_kv, esc_limit_a, battery_voltage_v, 
                       prop_diameter_in, prop_pitch_in, throttle_pct]])
        X_scaled = self.scaler.transform(X)

        # Predict all outputs
        predictions = {}
        for target_name, model in self.models.items():
            predictions[target_name] = model.predict(X_scaled)[0]

        return predictions

    def generate_test_report(self, motor_kv, esc_limit_a, battery_voltage_v,
                            prop_diameter_in, prop_pitch_in, prop_mfg='Unknown',
                            motor_mfg='Unknown', esc_mfg='Unknown',
                            throttle_levels=None):
        """
        Generate complete test report CSV for new configuration
        """
        if throttle_levels is None:
            throttle_levels = list(range(1000, 2050, 50))

        data = []
        for throttle_us in throttle_levels:
            throttle_pct = (throttle_us - 1000) / 10

            pred = self.predict(motor_kv, esc_limit_a, battery_voltage_v,
                               prop_diameter_in, prop_pitch_in, throttle_pct)

            row = {
                'Time (s)': len(data) * 2,  # Estimate ~2 seconds per measurement
                'Throttle (µs)': throttle_us,
                'Rotation speed (rpm)': int(pred.get('rpm', 0)),
                'Thrust (kgf)': pred.get('thrust_kgf', 0),
                'Torque (N⋅m)': pred.get('torque_nm', 0),
                'Voltage (V)': pred.get('voltage_v', battery_voltage_v),
                'Current (A)': pred.get('current_a', 0),
                'Electrical power (W)': pred.get('electrical_power_w', 0),
                'Motor manufacturer': motor_mfg,
                'Motor kv': motor_kv,
                'Propeller size(diameter)': prop_diameter_in,
                'Propeller size(pitch)': prop_pitch_in,
                'Propeller manufacturer': prop_mfg,
                'ESC manufacturer': esc_mfg,
                'ESC limit': esc_limit_a,
            }
            data.append(row)

        return pd.DataFrame(data)

    def save_models(self):
        """Save trained models to disk"""
        for name, model in self.models.items():
            joblib.dump(model, self.model_dir / f'{name}_model.pkl')
        joblib.dump(self.scaler, self.model_dir / 'scaler.pkl')
        print(f"Models saved to {self.model_dir}")

    def load_models(self):
        """Load pre-trained models from disk"""
        for model_file in self.model_dir.glob('*_model.pkl'):
            model_name = model_file.stem.replace('_model', '')
            self.models[model_name] = joblib.load(model_file)
        self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
        print(f"Loaded {len(self.models)} models")


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = PropulsionModelPipeline(model_dir='./propulsion_models')

    # Load experimental data
    # data = pipeline.load_experimental_data('./experimental_csv_files')

    # Train models
    # pipeline.train(data)
    # pipeline.save_models()

    # Load pre-trained models
    pipeline.load_models()

    # Generate prediction for new configuration
    print("\nGenerating test report for new configuration:")
    print("  Motor: T-Motor 2850KV")
    print("  ESC: Hobbywing 30A")
    print("  Battery: 3S 11.1V")
    print("  Propeller: APC 6x4")

    report = pipeline.generate_test_report(
        motor_kv=2850,
        esc_limit_a=30,
        battery_voltage_v=11.1,
        prop_diameter_in=6,
        prop_pitch_in=4,
        prop_mfg='APC',
        motor_mfg='T-Motor',
        esc_mfg='Hobbywing'
    )

    print("\nGenerated Report:")
    print(report.to_string())

    # Save report
    report.to_csv('test_report_example.csv', index=False)
    print("\nReport saved to test_report_example.csv")

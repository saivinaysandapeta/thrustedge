# Machine Learning Model Training for Propulsion Systems
## Data-Driven Surrogate Model for Motor-ESC-Battery-Propeller Combinations

---

## 1. Problem Definition

### What We're Solving
You have **300+ experimental CSV files** from actual thrust stand testing. Each file contains:
- **Inputs**: Motor KV, ESC Amperage Limit, Battery Voltage, Propeller Diameter, Propeller Pitch
- **Outputs**: RPM, Thrust, Power, Efficiency, Current, Voltage, Torque (at multiple throttle points)

**Goal**: Train a ML model to predict performance for ANY unseen combination of (Motor KV, ESC A, Battery V, Prop D, Prop P)

### Why This is Hard
1. **Nonlinear relationships**: RPM ∝ KV × V, but thrust ∝ RPM^2 × geometry
2. **Propeller geometry matters**: Same diameter/pitch ≠ same performance (airfoil profile, blade shape)
3. **Multi-output regression**: Predicting 7+ outputs simultaneously
4. **Generalization**: Model must work for combinations never tested

---

## 2. Recommended Approach

### Two-Stage Architecture

#### Stage 1: Physics-Based Feature Engineering
Extract aerodynamic/electrical features from raw parameters:
```
Inputs (5 parameters):
  - Motor KV (revolutions per volt)
  - ESC Limit (amperage)
  - Battery Voltage
  - Propeller Diameter (inches)
  - Propeller Pitch (inches)

Engineered Features (15-20 features):
  - Max RPM = KV × Battery Voltage
  - Propeller Disk Area = π × (Diameter/2)²
  - Pitch-to-Diameter Ratio = Pitch / Diameter
  - Max Electrical Power = Battery Voltage × ESC Limit
  - Advance Ratio (derived from operating point)
  - Thrust Coefficient (CT) ∝ Thrust / (density × RPM² × Area)
  - Power Coefficient (CP) ∝ Power / (density × RPM³ × Area)
  - etc.
```

#### Stage 2: Multi-Output Regression Model
```
Model Options (ranked by suitability):

1. **Gradient-Boosted Regression (XGBoost/LightGBM)** ★★★★★
   - Best for 300+ samples with mixed nonlinearity
   - Can handle multiple outputs with separate models
   - Interpretable feature importance
   - Fast inference (good for web app)
   
2. **Physics-Informed Neural Networks (PINNs)** ★★★★☆
   - Incorporates known physics constraints
   - Better generalization to unseen regions
   - More complex training
   
3. **Multi-Output Neural Network** ★★★★
   - Shared hidden layers between outputs
   - Captures correlations (RPM → Thrust → Power)
   - Moderate complexity

4. **Ensemble Methods** ★★★★
   - Combine multiple models
   - Uncertainty quantification
   - Highest robustness
```

### Recommended Stack
- **Language**: Python 3.10+
- **Data Prep**: Pandas, NumPy, scikit-learn
- **Model Training**: XGBoost + Neural Networks (TensorFlow)
- **Deployment**: FastAPI + JavaScript frontend
- **Output**: Generate CSV files matching your experimental format

---

## 3. Step-by-Step Training Pipeline

### Step 1: Data Collection & Standardization

**Input Format** (what you have):
```
Each CSV file = one experimental test run
Columns: throttle, rpm, thrust, power, efficiency, current, voltage, torque
Metadata in filename or separate: motor_kv, esc_a, battery_v, prop_d, prop_p
```

**Standardization Required**:
```python
1. Extract metadata from filenames/headers
   Example: "T-Motor_2850KV_30A_11.1V_6x4_APC.csv"
   → Motor KV: 2850
   → ESC Limit: 30A
   → Battery Voltage: 11.1V
   → Prop Diameter: 6"
   → Prop Pitch: 4"

2. Normalize all units (metric for ML)
   - Diameter/Pitch: inches → meters
   - Thrust: grams → kg
   - Power: Watts (already SI)
   - Voltage: Volts (SI)
   - Current: Amps (SI)

3. Aggregate data
   - 11 throttle points per file × 300 files = 3,300 training samples
   - More if you have more throttle points
```

**Code Structure**:
```python
import pandas as pd
import numpy as np
from pathlib import Path

class DataPreprocessor:
    def __init__(self, csv_folder):
        self.csv_folder = Path(csv_folder)
        self.data = []
    
    def extract_metadata(self, filename):
        """Extract motor KV, ESC A, Battery V, Prop D, Prop P from filename"""
        # Parse filename format: "manufacturer_kvKV_aA_vV_dxp_propname.csv"
        pass
    
    def standardize_units(self, df):
        """Convert all to SI units"""
        df['diameter_m'] = df['diameter_inches'] * 0.0254
        df['pitch_m'] = df['pitch_inches'] * 0.0254
        df['thrust_kg'] = df['thrust_grams'] / 1000
        return df
    
    def load_all_csvs(self):
        """Load and aggregate all CSV files"""
        for csv_file in self.csv_folder.glob('*.csv'):
            metadata = self.extract_metadata(csv_file.name)
            df = pd.read_csv(csv_file)
            df = self.standardize_units(df)
            # Add metadata columns
            for key, value in metadata.items():
                df[key] = value
            self.data.append(df)
        
        self.df = pd.concat(self.data, ignore_index=True)
        return self.df
```

---

### Step 2: Feature Engineering

**Physics-Based Features**:
```python
class FeatureEngineer:
    def __init__(self, df):
        self.df = df
    
    def engineer_features(self):
        """Create aerodynamic & electrical features"""
        
        # Electrical features
        self.df['max_rpm'] = self.df['motor_kv'] * self.df['battery_voltage']
        self.df['rpm_percentage'] = self.df['rpm'] / self.df['max_rpm']
        self.df['max_power_w'] = self.df['battery_voltage'] * self.df['esc_limit_a']
        
        # Propeller geometry features
        self.df['prop_area_m2'] = np.pi * (self.df['diameter_m']/2)**2
        self.df['pitch_diameter_ratio'] = self.df['pitch_m'] / self.df['diameter_m']
        self.df['aspect_ratio'] = self.df['diameter_m'] / (self.df['pitch_m'] + 0.001)
        
        # Aerodynamic coefficients (from experimental data)
        RHO = 1.225  # air density at sea level
        self.df['thrust_coeff'] = (
            self.df['thrust_kg'] * 9.81 / 
            (RHO * self.df['prop_area_m2'] * (self.df['rpm']/60)**2)
        )
        self.df['power_coeff'] = (
            self.df['power_w'] / 
            (RHO * self.df['prop_area_m2'] * (self.df['rpm']/60)**3)
        )
        
        # Operating conditions
        self.df['advance_ratio'] = 0  # Static thrust (no forward velocity)
        self.df['tip_speed_ms'] = np.pi * self.df['diameter_m'] * (self.df['rpm']/60)
        
        return self.df
```

**Feature List for Model**:
```
Inputs (5):
  1. motor_kv
  2. esc_limit_a
  3. battery_voltage
  4. prop_diameter_m
  5. prop_pitch_m

Engineered Features (20+):
  - max_rpm, rpm_percentage, max_power_w
  - prop_area_m2, pitch_diameter_ratio, aspect_ratio
  - thrust_coeff, power_coeff
  - tip_speed_ms
  - throttle_percentage (from test data)
  - interaction terms: kv × battery_v, diameter × pitch
```

---

### Step 3: Data Splitting & Normalization

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Separate inputs from outputs
X = df[['motor_kv', 'esc_limit_a', 'battery_voltage', 
         'prop_diameter_m', 'prop_pitch_m', 'throttle_percentage']]
y = df[['rpm', 'thrust_kg', 'power_w', 'efficiency', 
        'current_a', 'voltage_v', 'torque_nm']]

# 2. Normalize features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# 3. Normalize targets (IMPORTANT!)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# 4. Split data (80/20 with stratification by motor KV)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, 
    test_size=0.2,
    random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")
print(f"Outputs: {y_train.shape[1]}")
```

---

### Step 4: Model Training

#### Option A: Gradient-Boosted Regression (RECOMMENDED)

```python
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Train separate XGBoost model for each output
models = {}
output_names = ['rpm', 'thrust_kg', 'power_w', 'efficiency', 
                'current_a', 'voltage_v', 'torque_nm']

for output_name in output_names:
    # Get output column index
    idx = output_names.index(output_name)
    y_single = y_train[:, idx]
    
    # Train XGBoost
    model = XGBRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20
    )
    
    model.fit(
        X_train, y_single,
        eval_set=[(X_test, y_test[:, idx])],
        verbose=10
    )
    
    models[output_name] = model
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test[:, idx], y_pred))
    r2 = r2_score(y_test[:, idx], y_pred)
    
    print(f"{output_name}: RMSE={rmse:.4f}, R²={r2:.4f}")

# Save models
import pickle
with open('propulsion_models.pkl', 'wb') as f:
    pickle.dump({
        'models': models,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'output_names': output_names
    }, f)
```

**Expected Performance**:
- R² > 0.95 for RPM (strong relationship with inputs)
- R² > 0.90 for Thrust (good relationship with aerodynamics)
- R² > 0.85 for Efficiency (more variable, dependent on small RPM changes)

#### Option B: Multi-Output Neural Network

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_dim, output_dim):
    """Multi-output neural network"""
    
    inputs = layers.Input(shape=(input_dim,))
    
    # Shared hidden layers
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(32, activation='relu')(x)
    
    # Task-specific output heads
    rpm_out = layers.Dense(1, name='rpm')(x)
    thrust_out = layers.Dense(1, name='thrust')(x)
    power_out = layers.Dense(1, name='power')(x)
    efficiency_out = layers.Dense(1, name='efficiency')(x)
    current_out = layers.Dense(1, name='current')(x)
    voltage_out = layers.Dense(1, name='voltage')(x)
    torque_out = layers.Dense(1, name='torque')(x)
    
    model = keras.Model(
        inputs=inputs,
        outputs=[rpm_out, thrust_out, power_out, efficiency_out, 
                current_out, voltage_out, torque_out]
    )
    
    return model

# Build and compile
model = build_model(input_dim=X_train.shape[1], output_dim=7)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'rpm': 'mse',
        'thrust': 'mse',
        'power': 'mse',
        'efficiency': 'mse',
        'current': 'mse',
        'voltage': 'mse',
        'torque': 'mse'
    },
    metrics=['mae']
)

# Train
history = model.fit(
    X_train,
    {
        'rpm': y_train[:, 0],
        'thrust': y_train[:, 1],
        'power': y_train[:, 2],
        'efficiency': y_train[:, 3],
        'current': y_train[:, 4],
        'voltage': y_train[:, 5],
        'torque': y_train[:, 6]
    },
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
    ]
)

# Save
model.save('propulsion_model.keras')
```

---

### Step 5: Handling Unseen Propeller Geometry

**The Challenge**: Two propellers with same D/P might have different airfoil profiles.

**Solutions**:

1. **Add Propeller Manufacturer/Material as Features**
   ```python
   # Encode categorical features
   X['prop_manufacturer'] = pd.Categorical(X['prop_mfg']).codes
   X['prop_material'] = pd.Categorical(X['prop_material']).codes
   # This helps model learn material-specific efficiency patterns
   ```

2. **Use Airfoil Descriptors (if available)**
   ```
   If you have airfoil data:
   - Camber (curvature)
   - Thickness
   - Pitch angle distribution
   - Solidity (blade area / disk area)
   These should be measured or extracted from CAD models
   ```

3. **Create Propeller Family Embeddings**
   ```python
   # Group similar propellers, train family-specific models
   propeller_families = {
       'APC_Plastic': [samples with APC plastic props],
       'MAS_Carbon': [samples with MAS carbon props],
       'Custom_Wooden': [samples with wooden props]
   }
   # Train separate models per family, then ensemble
   ```

4. **Use Transfer Learning for New Geometries**
   ```python
   # Base model trained on known props
   # Fine-tune on small dataset of new propeller
   model_new_prop = keras.models.clone_model(model)
   model_new_prop.set_weights(model.get_weights())
   
   # Retrain last 2 layers with new propeller data
   for layer in model_new_prop.layers[:-2]:
       layer.trainable = False
   
   model_new_prop.fit(X_new_prop, y_new_prop, epochs=50)
   ```

---

## 4. Making Predictions for New Combinations

```python
import pickle
import numpy as np

def predict_performance(motor_kv, esc_limit_a, battery_v, 
                       prop_diameter_in, prop_pitch_in, 
                       throttle_percent=100):
    """
    Predict performance for new combination
    
    Args:
        motor_kv: Motor KV rating
        esc_limit_a: ESC current limit
        battery_v: Battery nominal voltage
        prop_diameter_in: Propeller diameter (inches)
        prop_pitch_in: Propeller pitch (inches)
        throttle_percent: Operating throttle (0-100%)
    
    Returns:
        dict with predicted: rpm, thrust, power, efficiency, etc.
    """
    
    # Load trained models
    with open('propulsion_models.pkl', 'rb') as f:
        pkg = pickle.load(f)
        models = pkg['models']
        scaler_X = pkg['scaler_X']
        scaler_y = pkg['scaler_y']
        output_names = pkg['output_names']
    
    # Convert units
    prop_diameter_m = prop_diameter_in * 0.0254
    prop_pitch_m = prop_pitch_in * 0.0254
    
    # Engineer features
    max_rpm = motor_kv * battery_v
    prop_area_m2 = np.pi * (prop_diameter_m/2)**2
    pitch_diameter_ratio = prop_pitch_m / prop_diameter_m
    
    # Create input vector
    X_new = np.array([[
        motor_kv,
        esc_limit_a,
        battery_v,
        prop_diameter_m,
        prop_pitch_m,
        throttle_percent
    ]])
    
    # Normalize
    X_new_scaled = scaler_X.transform(X_new)
    
    # Predict
    predictions = {}
    for output_name in output_names:
        model = models[output_name]
        y_pred_scaled = model.predict(X_new_scaled)
        # Denormalize (approximate - ideally use scaler)
        predictions[output_name] = y_pred_scaled[0][0]
    
    return predictions

# Example usage
result = predict_performance(
    motor_kv=2850,
    esc_limit_a=40,
    battery_v=14.8,
    prop_diameter_in=6,
    prop_pitch_in=4,
    throttle_percent=100
)

print(f"RPM: {result['rpm']:.0f}")
print(f"Thrust: {result['thrust_kg']:.3f} kg ({result['thrust_kg']*1000:.0f}g)")
print(f"Power: {result['power_w']:.1f} W")
print(f"Efficiency: {result['efficiency']:.1f} %")
```

---

## 5. Model Validation & Testing

```python
def validate_model(X_test, y_test, models, scaler_y, output_names):
    """Comprehensive model evaluation"""
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    results = {}
    
    for idx, output_name in enumerate(output_names):
        model = models[output_name]
        
        y_true = y_test[:, idx]
        y_pred = model.predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        results[output_name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        print(f"\n{output_name}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {results[output_name]['MAPE']:.2f}%")
    
    return results

# Acceptance Criteria
ACCEPTANCE_CRITERIA = {
    'rpm': {'R2': 0.95, 'MAPE': 5},        # ±5% error acceptable
    'thrust': {'R2': 0.90, 'MAPE': 8},     # ±8% error acceptable
    'power': {'R2': 0.88, 'MAPE': 10},     # ±10% error acceptable
    'efficiency': {'R2': 0.80, 'MAPE': 15}, # ±15% error acceptable (variable)
}
```

---

## 6. Deployment: Web Application

Your website workflow:
```
1. User inputs: Motor KV, ESC A, Battery V, Prop D, Prop P
2. Model predicts: RPM, Thrust, Power, Efficiency, Current, Voltage, Torque
3. Generate CSV at multiple throttle points (like your experimental format)
4. Download as CSV file
```

See separate "web_deployment.md" for implementation.

---

## 7. Addressing Propeller Geometry Variability

### Why Same D/P ≠ Same Performance

```
Propeller Geometry Parameters:
- Blade count (2, 3, 4 blades) → different solidity
- Airfoil profile (e.g., NACA 4-series) → different lift curves
- Twist distribution along span → different pitch angle at root/tip
- Thickness distribution → affects stiffness, drag
- Blade surface finish → affects boundary layer (Reynolds effects)
- Hub diameter ratio → affects efficiency
```

### Solutions in Your ML Model

**Option 1: Add Propeller Metadata**
```python
X = df[['motor_kv', 'esc_limit_a', 'battery_voltage', 
         'prop_diameter_m', 'prop_pitch_m',
         'prop_manufacturer',     # Add this!
         'prop_material',          # Add this!
         'blade_count',            # Add this!
         'prop_family'             # Add this!
]]
```

**Option 2: Propeller Encoding**
```python
# Create lookup table
PROPELLER_ENCODING = {
    'APC_6x4_plastic': {'solidity': 0.08, 'airfoil': 'NACA4415', 'twist_range': 25},
    'MAS_6x4_carbon': {'solidity': 0.10, 'airfoil': 'MAS7x6', 'twist_range': 30},
    'Custom_6x4_wood': {'solidity': 0.12, 'airfoil': 'custom', 'twist_range': 20},
}

# Add features
X['prop_solidity'] = X['prop_id'].map(lambda x: PROPELLER_ENCODING[x]['solidity'])
X['prop_twist_range'] = X['prop_id'].map(lambda x: PROPELLER_ENCODING[x]['twist_range'])
```

**Option 3: Hybrid Physics + ML (PINN approach)**
```python
# Use known physics constraints in loss function
def physics_loss(y_true, y_pred):
    """
    Ensure predictions respect physical laws
    e.g., Thrust = CT × density × Area × (RPM)²
    """
    ct_true, rho, area = extract_physics_params(y_true)
    ct_pred = calculate_ct_from_outputs(y_pred)
    
    physics_constraint_loss = |ct_pred - ct_true|²
    return mse_loss + 0.1 * physics_constraint_loss
```

---

## Summary: Recommended Implementation Path

1. ✅ **Collect & standardize** 300+ CSV files
2. ✅ **Engineer features** (physics-based)
3. ✅ **Train XGBoost** (7 separate models for each output)
4. ✅ **Add propeller metadata** (manufacturer, material, family)
5. ✅ **Validate** (R² > 0.90 for main outputs)
6. ✅ **Deploy** (FastAPI backend + React frontend)
7. ✅ **Generate CSV** with predictions at 11 throttle points

**Expected Accuracy**:
- RPM: ±3-5% error (very good)
- Thrust: ±5-8% error (acceptable)
- Power: ±8-10% error (acceptable)
- Efficiency: ±10-15% error (acceptable, more variable)

This approach balances complexity, interpretability, and prediction accuracy while handling the fundamental challenge: generalization to unseen propeller geometries through metadata encoding and physics constraints.

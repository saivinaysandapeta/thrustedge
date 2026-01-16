# ML Model Training: Complete Implementation Guide

## Quick Summary

You're building a **data-driven surrogate model** to predict propulsion system performance from electrical & mechanical parameters.

### What You Have
- 300+ experimental CSV files from actual thrust stand testing (with sensors)
- Each file: measurements at 11 throttle points (1000-2000 µs)
- Real data: RPM, Thrust, Power, Efficiency, Current, Voltage, Torque

### What You're Building
- **ML Model**: Maps (Motor KV, ESC A, Battery V, Prop Diameter, Prop Pitch) → (RPM, Thrust, Power, Efficiency, etc.)
- **Website**: User inputs parameters → Model predicts → Download CSV with 11 throttle points
- **Generalization**: Works for motor/prop combinations NOT in original 300 files

### Why Physics Matters
```
Electrical Domain:          Mechanical Domain:      Aerodynamic Domain:
Motor KV ─┐               RPM ─┐                   Thrust = f(RPM, Propeller)
Battery V ├─→ RPM ─→ Torque ─┼─→ Power ─┐
ESC A ────┘                   │          ├─→ Thrust, Efficiency
                              └─→ Power ─┘

Same Prop D/P ≠ Same Performance:
- Airfoil geometry differs (NACA 4415 vs custom)
- Blade twist distribution varies
- Material affects blade stiffness → efficiency
- Manufacturing tolerances matter

Solution: Add categorical features (propeller manufacturer, material, family)
```

---

## Implementation Checklist

### Phase 1: Data Preparation (Week 1)

- [ ] **Organize CSV files**
  - Create folder: `./thrust_test_data/`
  - Place all 300+ CSV files here
  - Verify format: `throttle, rpm, thrust, power, efficiency, current, voltage, torque, [metadata columns]`

- [ ] **Extract metadata**
  ```python
  # If metadata in filename: "T-Motor_2850KV_30A_11.1V_6x4_APC.csv"
  # Extract: motor_kv=2850, esc_limit_a=30, battery_voltage=11.1, 
  #         prop_diameter=6, prop_pitch=4
  
  # If metadata in CSV columns: ensure these columns exist
  # Column names: motor_kv, esc_limit_a, battery_voltage, 
  #             prop_diameter, prop_pitch
  ```

- [ ] **Check data quality**
  ```python
  import pandas as pd
  
  df = pd.read_csv('sample_file.csv')
  print(df.describe())  # Check for outliers
  print(df.isnull().sum())  # Check missing values
  print(df.dtypes)  # Verify data types
  ```

### Phase 2: Model Training (Week 2)

- [ ] **Install dependencies**
  ```bash
  pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn
  ```

- [ ] **Run training script**
  ```bash
  python train_propulsion_model.py
  # Outputs: propulsion_model.pkl (~50MB)
  ```

- [ ] **Validate model performance**
  ```
  Expected metrics (AFTER training):
  - RPM: R² > 0.95, RMSE < 5%
  - Thrust: R² > 0.90, RMSE < 8%
  - Power: R² > 0.88, RMSE < 10%
  - Efficiency: R² > 0.80, RMSE < 15%
  
  If metrics are poor:
  ✓ More training data needed
  ✓ Feature engineering improvements
  ✓ Model hyperparameter tuning
  ```

### Phase 3: Backend Deployment (Week 2-3)

- [ ] **Set up FastAPI**
  ```bash
  pip install fastapi uvicorn
  python backend.py
  # Server runs: http://localhost:8000
  ```

- [ ] **Test API endpoints**
  ```bash
  # Test health check
  curl http://localhost:8000/
  
  # Test prediction
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
      "motor_kv": 2850,
      "esc_limit_a": 30,
      "battery_voltage": 11.1,
      "prop_diameter_in": 6,
      "prop_pitch_in": 4,
      "num_points": 11
    }'
  ```

### Phase 4: Frontend Integration (Week 3)

- [ ] **Update HTML**
  - Replace `generateReport()` function (see web_deployment_guide.md)
  - Update button onclick handlers
  - Add ML results display section

- [ ] **Update JavaScript**
  - Modify `generateReport()` to call `/predict` API
  - Add `downloadCSVFromML()` function
  - Handle async/await for API calls

- [ ] **Test locally**
  ```bash
  # Terminal 1: Run backend
  python backend.py
  
  # Terminal 2: Serve frontend
  python -m http.server 8080
  
  # Browser: http://localhost:8080
  ```

### Phase 5: Production Deployment (Week 4)

- [ ] **Choose hosting**
  - Option A: Heroku (free tier or $5-7/month)
  - Option B: AWS Lambda + API Gateway
  - Option C: DigitalOcean App Platform
  - Option D: Replit (free for education)

- [ ] **Deploy**
  - See web_deployment_guide.md for detailed instructions
  - Set up environment variables
  - Test on production URL

---

## Addressing Propeller Geometry Variability

### Problem
Same propeller D/P can have vastly different performance depending on:
- Airfoil profile (NACA, custom geometry)
- Blade count (2, 3, 4 blades)
- Pitch distribution (how pitch changes along blade span)
- Material (plastic, carbon, wood) affects stiffness
- Surface finish affects drag

### Solution: Metadata Encoding

**Step 1: Add Propeller Descriptors to Dataset**

```python
# Enhanced CSV schema
columns = [
    # Original data
    'throttle', 'rpm', 'thrust', 'power', 'efficiency', 'current', 'voltage', 'torque',
    
    # System parameters
    'motor_kv', 'esc_limit_a', 'battery_voltage',
    'prop_diameter', 'prop_pitch',
    
    # PROPELLER METADATA (add these!)
    'prop_manufacturer',    # 'APC', 'MAS', 'Graupner', 'Custom'
    'prop_material',        # 'Plastic', 'Carbon', 'Wood'
    'prop_blade_count',     # 2, 3, 4
    'prop_family',          # 'MAS3D', 'APC_Sport', etc.
    'prop_batch',           # Distinguish test batches
]
```

**Step 2: Extract from Metadata**

```python
# In train_propulsion_model.py
class FeatureEngineer:
    def engineer_all_features(self, df):
        # ... existing features ...
        
        # Add propeller metadata encoding
        propeller_types = df['prop_manufacturer'].unique()
        for prop_type in propeller_types:
            mask = df['prop_manufacturer'] == prop_type
            df[f'is_{prop_type}'] = mask.astype(int)
        
        materials = df['prop_material'].unique()
        for material in materials:
            mask = df['prop_material'] == material
            df[f'material_{material}'] = mask.astype(int)
        
        return df
```

**Step 3: Stratified Training**

```python
from sklearn.model_selection import train_test_split

# Split by propeller family to test generalization
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled,
    test_size=0.2,
    random_state=42,
    stratify=df['prop_manufacturer']  # Ensure balanced distribution
)

# Evaluate on unseen propeller families
for family in df['prop_manufacturer'].unique():
    mask = df['prop_manufacturer'] == family
    X_family = X_scaled[mask]
    y_family = y_scaled[mask]
    
    predictions = model.predict(X_family)
    r2 = r2_score(y_family, predictions)
    print(f"Family {family}: R² = {r2:.4f}")
```

**Step 4: Transfer Learning for New Propellers**

```python
# When you have a new propeller type with 10-20 test points:
new_prop_data = load_new_propeller_dataset()

# Fine-tune existing model on new propeller data
model_adapted = clone(existing_model)

# Freeze most layers, only train on new data
# (XGBoost: retrain last 20% of trees)
model_adapted.fit(
    new_prop_data['X'],
    new_prop_data['y'],
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    xgb_model=existing_model.get_booster()  # Use pre-trained weights
)
```

---

## Troubleshooting Guide

### Problem 1: Model Performance is Poor (R² < 0.80)

**Symptom**: Predictions don't match experimental data

**Root Causes & Solutions**:

1. **Insufficient data variability**
   - Check: All 300 files test different parameter combinations?
   - Fix: Ensure coverage across KV range (450-5000), ESC (5-300A), Voltage (7-48V)
   - Target: At least 10+ combinations per parameter region

2. **Data quality issues**
   - Check: Outliers, missing values, wrong units
   - Fix:
     ```python
     # Remove outliers
     for col in y.columns:
         mean, std = y[col].mean(), y[col].std()
         mask = (y[col] > mean - 3*std) & (y[col] < mean + 3*std)
         df = df[mask]
     
     # Check for duplicate measurements
     print(df.duplicated(subset=['motor_kv', 'esc_limit_a', 'battery_voltage', 
                                  'prop_diameter', 'prop_pitch', 'throttle']).sum())
     ```

3. **Feature engineering too simple**
   - Add propeller metadata (manufacturer, material, blade_count)
   - Add aerodynamic features (thrust_coefficient, power_coefficient)
   - Add interaction terms (kv × voltage, diameter × pitch)

4. **Wrong model hyperparameters**
   - Try different XGBoost settings:
     ```python
     # Grid search
     params = {
         'n_estimators': [100, 200, 300],
         'max_depth': [5, 7, 9],
         'learning_rate': [0.05, 0.1, 0.2],
     }
     # Evaluate each combination
     ```

### Problem 2: Model Overfits (High R² on train, low on test)

**Symptom**: Predictions work on training data but fail on new combinations

**Solutions**:

```python
# 1. Increase test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42  # Increase from 0.2 to 0.3
)

# 2. Add regularization
model = XGBRegressor(
    n_estimators=200,
    max_depth=7,
    subsample=0.7,        # Use 70% of samples per tree
    colsample_bytree=0.7, # Use 70% of features per tree
    reg_lambda=1.0,       # L2 regularization
    reg_alpha=0.1,        # L1 regularization
)

# 3. Cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(f"CV scores: {scores.mean():.4f} ± {scores.std():.4f}")

# 4. Simpler model
# Try linear regression or shallow trees first
from sklearn.linear_model import Ridge
model_simple = Ridge(alpha=1.0)
```

### Problem 3: Predictions Have Negative/Impossible Values

**Symptom**: Model predicts negative RPM or >100% efficiency

**Solution**:

```python
# Post-processing: Enforce physical constraints
def enforce_physical_constraints(predictions):
    predictions['rpm'] = np.maximum(0, predictions['rpm'])
    predictions['thrust_kg'] = np.maximum(0, predictions['thrust_kg'])
    predictions['power_w'] = np.maximum(0, predictions['power_w'])
    predictions['efficiency'] = np.clip(predictions['efficiency'], 0, 100)
    predictions['current_a'] = np.maximum(0, predictions['current_a'])
    predictions['torque_nm'] = np.maximum(0, predictions['torque_nm'])
    
    # Consistency checks
    # RPM should be proportional to KV × Voltage
    max_rpm_expected = predictions['motor_kv'] * predictions['battery_voltage']
    if predictions['rpm'] > max_rpm_expected:
        predictions['rpm'] = max_rpm_expected * 0.95
    
    return predictions
```

### Problem 4: API Returns Error "Model not loaded"

**Symptom**: `/predict` endpoint returns 500 error

**Solution**:

```bash
# 1. Check model file exists
ls -la propulsion_model.pkl

# 2. Verify model file is readable
python -c "import pickle; pickle.load(open('propulsion_model.pkl', 'rb'))"

# 3. Check FastAPI startup logs
python backend.py 2>&1 | grep -i "model"

# 4. Run training first
python train_propulsion_model.py
```

### Problem 5: CSV Download is Corrupted

**Symptom**: Downloaded file is empty or malformed

**Solution**:

```python
# Test CSV generation locally
from backend import download_csv
from train_propulsion_model import PropulsionInput

test_input = PropulsionInput(
    motor_kv=2850,
    esc_limit_a=30,
    battery_voltage=11.1,
    prop_diameter_in=6,
    prop_pitch_in=4
)

result = download_csv(test_input)
print(result)
```

---

## Quick Start Template

If you're starting from scratch:

```bash
# 1. Create project structure
mkdir thrusteedge-ml
cd thrusteedge-ml
mkdir thrust_test_data
mkdir models

# 2. Create Python virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install pandas numpy scikit-learn xgboost fastapi uvicorn

# 4. Copy training script
cp train_propulsion_model.py .
cp backend.py .

# 5. Place your CSV files
cp /path/to/csv/files/*.csv thrust_test_data/

# 6. Train model
python train_propulsion_model.py

# 7. Run API server
python backend.py

# 8. Test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"motor_kv": 2850, "esc_limit_a": 30, "battery_voltage": 11.1, "prop_diameter_in": 6, "prop_pitch_in": 4}'
```

---

## Key Insights for Success

1. **Data Quality > Model Complexity**
   - 300 high-quality samples beats 3000 noisy samples
   - Verify every CSV file is properly formatted

2. **Physics Constraints Matter**
   - RPM ∝ KV × Voltage (basic electricity)
   - Thrust ∝ RPM² × Propeller Area (momentum theory)
   - Efficiency ∝ (Mechanical Power / Electrical Power)

3. **Propeller Geometry is Crucial**
   - Same D/P, different airfoil = different performance
   - Add manufacturer/material/batch as features

4. **Test on Unseen Combinations**
   - If model trained on KV [900, 1400, 2200, 2850]
   - Test on KV 1600 (interpolation)
   - Test on KV 3500 (extrapolation - expect lower accuracy)

5. **Document Your Assumptions**
   - Record how propeller diameter/pitch were measured
   - Note if thrust stand was calibrated
   - Track any data anomalies

This approach will give you a production-ready ML system that can predict propulsion performance for any motor/ESC/battery/propeller combination with ~5-10% accuracy.

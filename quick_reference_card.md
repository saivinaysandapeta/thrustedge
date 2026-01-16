# Quick Reference Card - Commands & Code Snippets

## 🚀 Installation & Setup (Copy-Paste Ready)

```bash
# 1. CREATE PROJECT STRUCTURE
mkdir -p propulsion_ml_system/{data/raw,data/processed,models,src,web_app/templates,notebooks,logs}
cd propulsion_ml_system

# 2. CREATE VIRTUAL ENVIRONMENT
python -m venv venv

# 3. ACTIVATE ENVIRONMENT
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. INSTALL ALL DEPENDENCIES
pip install pandas==2.0.3 numpy==1.24.3 scikit-learn==1.3.0 xgboost==2.0.0 joblib==1.3.1 openpyxl==3.1.2 flask==2.3.2 matplotlib==3.7.2 seaborn==0.12.2

# 5. VERIFY INSTALLATION
python -c "import pandas, xgboost, sklearn, flask; print('✓ All dependencies installed')"

# 6. PLACE YOUR 300+ EXCEL FILES
# Copy all files to: data/raw/
# Verify:
ls data/raw/*.xlsx | wc -l
```

---

## 📚 Code Snippets for Each Phase

### Phase 1: Quick Training Script
**File: `quick_train.py`**

```python
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load data
data_dir = Path("data/raw")
files = list(data_dir.glob("*.xlsx"))[:20]  # Start with 20
dfs = [pd.read_excel(f) for f in files if f.exists()]
df = pd.concat(dfs, ignore_index=True)

# Clean
numeric_cols = df.select_dtypes(include=[np.number]).columns
df = df[numeric_cols].dropna()

# Split data
X = df.drop(columns=['Rotation speed (rpm)', 'Thrust (kgf)'], errors='ignore')
y = df[['Rotation speed (rpm)', 'Thrust (kgf)']].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale & train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
score = model.score(X_test_scaled, y_test)
print(f"✓ Model trained - R² Score: {score:.4f}")

# Save
import os
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/quick_model.pkl")
joblib.dump(scaler, "models/quick_scaler.pkl")
joblib.dump(X.columns.tolist(), "models/quick_features.pkl")
print("✓ Models saved")
```

### Phase 2: Data Loader
**File: `src/data_loader.py` (Essential functions)**

```python
import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, raw_dir="data/raw"):
        self.raw_dir = Path(raw_dir)
    
    def load_all(self):
        files = list(self.raw_dir.glob("*.xlsx"))
        dfs = []
        for file in files:
            try:
                df = pd.read_excel(file)
                df['source'] = file.stem
                dfs.append(df)
            except:
                pass
        return pd.concat(dfs, ignore_index=True)
    
    def clean(self, df):
        df = df.dropna(how='all')
        df = df.drop_duplicates()
        return df

# Usage
loader = DataLoader()
df = loader.load_all()
df = loader.clean(df)
df.to_csv("data/processed/combined.csv", index=False)
```

### Phase 3: Feature Engineering
**File: `src/feature_engineering.py` (Key features)**

```python
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
    
    def create_features(self):
        df = self.df
        
        # Electrical
        if 'Voltage (V)' in df.columns and 'Current (A)' in df.columns:
            df['electrical_power'] = df['Voltage (V)'] * df['Current (A)']
        
        # Mechanical
        if 'Rotation speed (rpm)' in df.columns and 'Mechanical power (W)' in df.columns:
            omega = df['Rotation speed (rpm)'] * 2 * np.pi / 60
            df['motor_torque'] = df['Mechanical power (W)'] / omega.clip(lower=0.1)
        
        # Aerodynamic
        if 'Thrust (kgf)' in df.columns and 'propeller_diameter_inch' in df.columns:
            dia_m = df['propeller_diameter_inch'] * 0.0254
            area = np.pi * (dia_m / 2) ** 2
            df['disk_loading'] = (df['Thrust (kgf)'] * 9.81) / area.clip(lower=0.01)
        
        # Propeller tip speed
        if 'Rotation speed (rpm)' in df.columns and 'propeller_diameter_inch' in df.columns:
            dia_m = df['propeller_diameter_inch'] * 0.0254
            df['tip_speed'] = (df['Rotation speed (rpm)'] * dia_m * np.pi) / 60
        
        self.df = df
        return df

# Usage
fe = FeatureEngineer(df)
df_engineered = fe.create_features()
df_engineered.to_csv("data/processed/engineered.csv", index=False)
```

### Phase 4: Model Training
**File: `src/model_trainer.py` (Simplified)**

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv("data/processed/engineered.csv")

# Prepare
numeric = df.select_dtypes(include=[np.number]).columns.tolist()
targets = ['Rotation speed (rpm)', 'Thrust (kgf)']
features = [c for c in numeric if c not in targets]

X = df[features].fillna(df[features].median())
y = df[targets].fillna(0)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost for each target
models = {}
for target in targets:
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_train_scaled, y_train[target])
    score = model.score(X_test_scaled, y_test[target])
    print(f"{target}: R² = {score:.4f}")
    models[target] = model

# Save
joblib.dump(models, "models/models.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(features, "models/features.pkl")
print("✓ Models saved")
```

### Phase 5: Flask Web App
**File: `web_app/app.py` (Minimal version)**

```python
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load models
models = joblib.load("../models/models.pkl")
scaler = joblib.load("../models/scaler.pkl")
features = joblib.load("../models/features.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Create input
        input_dict = {f: float(data.get(f, 0)) for f in features}
        X = pd.DataFrame([input_dict])
        X_scaled = scaler.transform(X)
        
        # Predict
        predictions = {}
        for target, model in models.items():
            predictions[target] = float(model.predict(X_scaled)[0])
        
        return jsonify({'status': 'success', 'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

---

## 🎯 Essential Python Commands

### Data Exploration
```python
import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/combined.csv")

# Summary
print(df.info())
print(df.describe())
print(df.shape)  # (rows, columns)

# Missing values
print(df.isnull().sum())

# Data types
print(df.dtypes)

# Unique values
print(df['column_name'].unique())
print(df['column_name'].nunique())

# Value counts
print(df['column_name'].value_counts())

# Correlation
print(df.corr())
```

### Data Cleaning
```python
# Remove NaN
df_clean = df.dropna()
df_clean = df.dropna(subset=['col1', 'col2'])
df_clean = df.fillna(df.mean())

# Remove duplicates
df_clean = df.drop_duplicates()

# Select columns
df_subset = df[['col1', 'col2', 'col3']]
df_subset = df.select_dtypes(include=[np.number])

# Filter rows
df_filtered = df[df['column'] > value]
df_filtered = df[df['column'].isin(['value1', 'value2'])]

# Sort
df_sorted = df.sort_values(by='column')
```

### Model Training
```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_test = scaler.transform(X_test)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = xgb.XGBRegressor(n_estimators=200, max_depth=6)
model.fit(X_train, y_train)

# Evaluation
score = model.score(X_test, y_test)  # R² score
predictions = model.predict(X_test)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Mean CV Score: {cv_scores.mean():.4f}")

# Feature importance
importances = model.feature_importances_
```

### Model Saving/Loading
```python
import joblib

# Save
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "features.pkl")

# Load
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# Predict with loaded model
X_scaled = scaler.transform(X_new)
predictions = model.predict(X_scaled)
```

---

## 🔧 Troubleshooting Quick Fixes

### Issue: File not found
```python
from pathlib import Path
file = Path("data/raw/test.xlsx")
if file.exists():
    df = pd.read_excel(file)
else:
    print(f"File not found: {file}")
```

### Issue: Memory error with large dataset
```python
# Process in chunks
chunk_size = 1000
chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

# Or reduce to numeric columns only
df_numeric = df.select_dtypes(include=[np.number])
```

### Issue: NaN in predictions
```python
# Check for missing values
print("NaN values:", df.isnull().sum())

# Fill NaN
df = df.fillna(df.median())

# Check for infinity
print("Infinity values:", np.isinf(df).sum())

# Remove infinity
df = df[~np.isinf(df).any(axis=1)]
```

### Issue: Model not improving
```python
# Check data quality
print(df.describe())

# Verify scaling
print(X_scaled.mean(), X_scaled.std())  # Should be ~0 and ~1

# Try different hyperparameters
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1]
}

# Use grid search
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(xgb.XGBRegressor(), params, cv=5)
gs.fit(X_train, y_train)
print(gs.best_params_)
```

---

## 📊 Expected Output Examples

### After Phase 1 (Quick Training)
```
=====================================================
QUICK PROPULSION SYSTEM MODEL TRAINING
=====================================================

[1/5] Loading Excel files...
Found 300 files
  Loaded 5 files...
  Loaded 10 files...
✓ Combined 12500 rows

[2/5] Cleaning data...
✓ Cleaned to 12000 rows

[3/5] Creating features...
✓ Created 45 features

[4/5] Training model...
✓ Model trained - R² Score: 0.9125

[5/5] Saving model...
✓ Model saved to models/

============================================================
✓ TRAINING COMPLETE!
============================================================

Model Performance: 91.3%
Training samples: 9600
Test samples: 2400

Next: Run web_app/app.py to start the web interface
```

### Web Application Response
```json
{
  "status": "success",
  "input": {
    "motor_kv": 2850,
    "esc_amperage": 30,
    "battery_voltage": 7.4,
    "propeller_diameter": 7,
    "propeller_pitch": 6
  },
  "predictions": {
    "Rotation speed (rpm)": 8534,
    "Thrust (kgf)": 0.185,
    "Electrical power (W)": 22.5,
    "Motor & ESC efficiency (%)": 65.2,
    "Propeller efficiency (gf/W)": 5.81
  }
}
```

---

## 📝 File Naming Conventions

### Excel Files in data/raw/
```
Format: test_XXX_motorKV_propDIA_PITCH_escA_battV.xlsx

Examples:
- test_001_motor2850_prop5_3_esc30_batt7p4.xlsx
- test_002_motor1450_prop7_6_esc40_batt11p1.xlsx
- test_003_motor3520_prop8_5_esc50_batt14p8.xlsx

Components:
- Motor Kv: 1100, 1450, 1750, 2200, 2850, 3520, 4600
- Prop Dia: 5, 6, 7, 8, 9, 10, 12 (inches)
- Prop Pitch: 2.5, 3, 4, 5, 6, 7, 8 (inches)
- ESC A: 20, 30, 40, 50, 60, 80, 100, 120 (Amperes)
- Battery V: 3p7, 7p4, 11p1, 14p8, 22p2 (Voltage)
```

---

## 🚀 From Quick Start to Full System (Timeline)

```
Day 1: Quick Start (2-3 hours)
├─ Setup environment: 15 min
├─ Copy 300+ Excel files: 30 min
├─ Run quick_train.py: 30 min
├─ Launch Flask app: 15 min
└─ Test predictions: 30 min
   Result: Working prototype

Day 2: Data Pipeline (3-4 hours)
├─ Implement DataLoader: 60 min
├─ Parse filenames: 45 min
├─ Combine & clean data: 45 min
└─ Save processed dataset: 30 min
   Result: Clean 10k-30k row dataset

Day 3: Feature Engineering (3-4 hours)
├─ Create electrical features: 60 min
├─ Create mechanical features: 60 min
├─ Create aerodynamic features: 60 min
└─ Handle missing values & select: 45 min
   Result: 30-50 physics-informed features

Day 4: Professional Training (2-3 hours)
├─ Multi-output regression: 45 min
├─ Train & cross-validate: 60 min
├─ Save all models: 30 min
└─ Generate metrics report: 30 min
   Result: R² Score 0.92-0.95

Day 5: Web Interface (2-3 hours)
├─ Polish Flask backend: 60 min
├─ Improve HTML/CSS: 45 min
├─ Add batch processing: 45 min
└─ Testing & debugging: 30 min
   Result: Production-ready interface

Total: ~2 weeks for full system
```

---

## 💡 Pro Tips

1. **Start Small**: Process first 50 files, then scale to 300+
2. **Monitor Memory**: Check RAM usage while loading files
3. **Save Intermediate Results**: Save after each phase
4. **Test Early**: Try predictions with incomplete data
5. **Version Control**: Save model versions with dates
6. **Log Everything**: Keep prediction logs for analysis
7. **Regular Retraining**: Update models as new data arrives
8. **Validate Physics**: Check predictions make physical sense

---

## 📞 Quick Debugging Checklist

- [ ] All 300+ Excel files in data/raw/
- [ ] Python virtual environment activated
- [ ] All packages installed: `pip list | grep pandas`
- [ ] At least 4GB free RAM
- [ ] Models load correctly: `python -c "import joblib; joblib.load('models/quick_model.pkl')"`
- [ ] Web app starts: `python web_app/app.py`
- [ ] Accessible at: http://localhost:5000
- [ ] Form submission works
- [ ] CSV export works
- [ ] Predictions are reasonable (positive, within expected range)

Once all checked, you're ready to scale to production!


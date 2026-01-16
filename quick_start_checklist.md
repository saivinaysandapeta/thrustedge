# Quick-Start Implementation Checklist

## 🚀 Quick Start (5 minutes)

### Step 1: Project Setup
```bash
# Create project structure
mkdir -p propulsion_ml_system/{data/raw,data/processed,models,src,web_app/templates,notebooks,logs}
cd propulsion_ml_system

# Create Python virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn xgboost joblib openpyxl flask matplotlib seaborn

# Verify installation
python -c "import pandas, xgboost, flask; print('✓ All packages installed')"
```

### Step 2: Organize Data
```bash
# Copy all 300+ Excel files to data/raw/
# Rename files consistently:
# Format: test_XXX_motorKV_propDIAM_PITCH_escA_battVp0.xlsx
# Example: test_001_motor2850_prop7_6_esc30_batt7p4.xlsx
# (Use 'p' instead of '.' for voltage: 7p4 = 7.4V)

# Count files to verify
ls data/raw/*.xlsx | wc -l
```

### Step 3: Start Simple Training
```bash
# Navigate to project root
cd propulsion_ml_system

# Create quick training script
cat > quick_train.py << 'EOF'
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("QUICK PROPULSION SYSTEM MODEL TRAINING")
print("=" * 60)

# Step 1: Load and combine all Excel files
print("\n[1/5] Loading Excel files...")
data_dir = Path("data/raw")
excel_files = list(data_dir.glob("*.xlsx"))
print(f"Found {len(excel_files)} files")

dataframes = []
for i, file in enumerate(excel_files[:10]):  # Start with first 10
    try:
        df = pd.read_excel(file)
        df['source_file'] = file.stem
        dataframes.append(df)
        if (i + 1) % 5 == 0:
            print(f"  Loaded {i + 1} files...")
    except Exception as e:
        print(f"  ✗ Error loading {file.name}: {e}")

combined_df = pd.concat(dataframes, ignore_index=True)
print(f"✓ Combined {len(combined_df)} rows")

# Step 2: Clean data
print("\n[2/5] Cleaning data...")
# Remove rows where key columns are NaN
key_cols = ['Rotation speed (rpm)', 'Thrust (kgf)', 'Voltage (V)', 'Current (A)']
combined_df = combined_df[combined_df[key_cols].notna().all(axis=1)]
combined_df = combined_df.drop_duplicates()
print(f"✓ Cleaned to {len(combined_df)} rows")

# Step 3: Create features
print("\n[3/5] Creating features...")
# Simple features from existing columns
numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
X = combined_df[numeric_cols].fillna(combined_df[numeric_cols].median())

# Remove target from features
targets = ['Rotation speed (rpm)', 'Thrust (kgf)']
feature_cols = [col for col in X.columns if col not in targets]
X = X[feature_cols]

print(f"✓ Created {len(feature_cols)} features")

# Step 4: Train model
print("\n[4/5] Training model...")
y = combined_df[['Rotation speed (rpm)', 'Thrust (kgf)']].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train simple Random Forest
model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
model.fit(X_train_scaled, y_train)

score = model.score(X_test_scaled, y_test)
print(f"✓ Model trained - R² Score: {score:.4f}")

# Step 5: Save model
print("\n[5/5] Saving model...")
import os
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/quick_model.pkl")
joblib.dump(scaler, "models/quick_scaler.pkl")
joblib.dump(feature_cols, "models/quick_features.pkl")
print("✓ Model saved to models/")

print("\n" + "=" * 60)
print("✓ TRAINING COMPLETE!")
print("=" * 60)
print(f"\nModel Performance: {score:.1%}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print("\nNext: Run web_app/app.py to start the web interface")

EOF

python quick_train.py
```

### Step 4: Launch Web Application
```bash
# Create Flask app
mkdir -p web_app/templates
cat > web_app/app.py << 'EOF'
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__, template_folder='templates')

# Load model
try:
    model = joblib.load("../models/quick_model.pkl")
    scaler = joblib.load("../models/quick_scaler.pkl")
    features = joblib.load("../models/quick_features.pkl")
    print("✓ Model loaded")
except:
    model = None
    print("✗ Model not found - run quick_train.py first")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        
        # Create input array
        input_dict = {f: float(data.get(f, 0)) for f in features}
        X = np.array([list(input_dict.values())])
        X_scaled = scaler.transform(X)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        
        return jsonify({
            'rpm': float(prediction[0]),
            'thrust': float(prediction[1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

EOF

cat > web_app/templates/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Propulsion System Predictor</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 8px; }
        h1 { color: #333; }
        input { padding: 10px; margin: 10px 0; width: 100%; }
        button { padding: 12px 24px; background: #0066cc; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0052a3; }
        .results { margin-top: 20px; padding: 15px; background: white; border-radius: 4px; display: none; }
        .results.show { display: block; }
        .metric { margin: 10px 0; }
        .metric-label { font-weight: bold; color: #666; }
        .metric-value { font-size: 24px; color: #0066cc; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚁 Propulsion System Predictor</h1>
        <p>Predict motor RPM and thrust from component specifications</p>
        
        <form id="predictForm">
            <input type="number" placeholder="Motor Kv (e.g., 2850)" id="kv" required>
            <input type="number" placeholder="Battery Voltage (e.g., 7.4)" id="voltage" required>
            <input type="number" placeholder="Current (e.g., 5.0)" id="current" required>
            <input type="number" placeholder="Propeller Diameter (inch)" id="prop_dia" required>
            <input type="number" placeholder="Propeller Pitch (inch)" id="prop_pitch" required>
            <button type="submit">Predict Performance</button>
        </form>
        
        <div class="results" id="results">
            <div class="metric">
                <div class="metric-label">Predicted RPM:</div>
                <div class="metric-value" id="rpmValue">-</div>
            </div>
            <div class="metric">
                <div class="metric-label">Predicted Thrust (kgf):</div>
                <div class="metric-value" id="thrustValue">-</div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const data = {
                kv: document.getElementById('kv').value,
                voltage: document.getElementById('voltage').value,
                current: document.getElementById('current').value,
                prop_dia: document.getElementById('prop_dia').value,
                prop_pitch: document.getElementById('prop_pitch').value,
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                document.getElementById('rpmValue').textContent = result.rpm.toFixed(0);
                document.getElementById('thrustValue').textContent = result.thrust.toFixed(4);
                document.getElementById('results').classList.add('show');
            } catch (error) {
                alert('Prediction failed: ' + error);
            }
        });
    </script>
</body>
</html>
EOF

# Run app
cd web_app
python app.py
```

## 🎯 Implementation Phases (In Order)

### Phase 1: Quick Start (Today - 30 minutes)
- [ ] Setup virtual environment
- [ ] Install packages
- [ ] Copy 300+ Excel files to data/raw/
- [ ] Run quick_train.py
- [ ] Launch Flask app on localhost:5000
- [ ] Test single prediction

### Phase 2: Proper Data Pipeline (1-2 hours)
- [ ] Implement DataLoader class
- [ ] Parse filenames for component metadata
- [ ] Create combined_dataset.csv
- [ ] Data validation and cleaning
- [ ] Save processed data

### Phase 3: Physics-Informed Features (2-3 hours)
- [ ] Implement FeatureEngineer class
- [ ] Create electrical domain features (power, efficiency)
- [ ] Create mechanical domain features (torque, back-EMF)
- [ ] Create aerodynamic features (disk loading, tip speed)
- [ ] Create system coupling features
- [ ] Handle missing values
- [ ] Save engineered_features.csv

### Phase 4: Model Training (2-3 hours)
- [ ] Implement multi-output regression
- [ ] Train XGBoost models
- [ ] Cross-validation (5-fold)
- [ ] Evaluate performance
- [ ] Save trained models and scalers

### Phase 5: Professional Web Interface (2-3 hours)
- [ ] Build Flask backend with proper APIs
- [ ] Create modern HTML/CSS frontend
- [ ] Single prediction interface
- [ ] CSV batch prediction
- [ ] Results export to CSV/JSON
- [ ] Performance monitoring

### Phase 6: Advanced Features (Optional - 3-4 hours)
- [ ] Ensemble methods (XGBoost + RandomForest + Gradient Boosting)
- [ ] Uncertainty quantification
- [ ] Domain physics validation
- [ ] Model performance dashboard
- [ ] Automated retraining pipeline

---

## 📊 Expected Results

After Phase 1 (Quick Start):
```
✓ Training samples: ~1000-2000
✓ Model R² Score: 0.85-0.92
✓ Response time: <100ms
✓ Web interface running on localhost:5000
```

After Phase 3 (Full Implementation):
```
✓ Training samples: ~10,000-30,000 (all 300+ files)
✓ Model R² Score: 0.92-0.97
✓ Features: 30-50 physics-informed
✓ Outputs: RPM, Thrust, Power, Efficiency, Propeller Efficiency
✓ Batch processing: 100 predictions in <5 seconds
```

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'xgboost'"
```bash
pip install xgboost
pip install scikit-learn
```

### Issue: Excel files not loading
```python
# Verify file format
from openpyxl import load_workbook
wb = load_workbook('data/raw/test_001.xlsx')
print(wb.sheetnames)  # Should show sheet names
```

### Issue: Model not predicting correctly
```python
# Check data scaling
import numpy as np
print("Feature ranges:")
print(X.describe())

print("\nTarget ranges:")
print(y.describe())

# Data should be roughly in similar ranges
```

### Issue: Web app not loading on localhost:5000
```bash
# Check if port 5000 is in use
netstat -an | grep 5000  # Linux/Mac
netstat -ano | findstr :5000  # Windows

# Use different port
python app.py  # Change port=5000 to port=8000
```

---

## 💾 File Organization After Setup

```
propulsion_ml_system/
├── data/
│   ├── raw/                    # Your 300+ Excel files
│   │   ├── test_001.xlsx
│   │   ├── test_002.xlsx
│   │   └── ... (300+ files)
│   └── processed/
│       ├── combined_dataset.csv
│       └── engineered_features.csv
├── models/
│   ├── quick_model.pkl         # Phase 1 simple model
│   ├── rpm_predictor.pkl       # Phase 4 final models
│   ├── thrust_predictor.pkl
│   ├── power_predictor.pkl
│   ├── feature_scaler.pkl
│   ├── output_scalers.pkl
│   ├── feature_columns.pkl
│   └── output_columns.pkl
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model_trainer.py
│   ├── model_predictor.py
│   └── utils.py
├── web_app/
│   ├── app.py
│   ├── requirements.txt
│   └── templates/
│       └── index.html
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_training.ipynb
├── logs/
│   └── predictions.log
├── quick_train.py              # Phase 1 quick script
├── config.py
└── README.md
```

---

## 🎓 Learning Resources

If unfamiliar with concepts:

### Machine Learning Basics
- Regression: predicting continuous values (RPM, Thrust)
- Feature scaling: normalize inputs to 0-1 range
- Train/Test split: 80% training, 20% validation
- Cross-validation: evaluate model stability

### Physics Background for Your Models
- **Electrical Domain**: V × I = Power (Watts)
- **Mechanical Domain**: P = τ × ω (Power = Torque × Angular Velocity)
- **Aerodynamic Domain**: Thrust ∝ ρ × n² × D⁴ (Thrust coefficient)
- **Efficiency**: Motor_eff = Mechanical_Power / Electrical_Power

### Key Algorithms
- **XGBoost**: Gradient boosting - powerful for tabular data
- **Random Forest**: Ensemble - good for complex relationships
- **Neural Networks**: For non-linear patterns
- **Feature Scaling**: StandardScaler or MinMaxScaler

---

## 🚀 Deployment Options After Development

### Option 1: Local Desktop (Current)
```bash
python web_app/app.py
# Access: http://localhost:5000
```

### Option 2: Cloud Deployment (AWS/Google Cloud)
```bash
# Docker containerization
docker build -t propulsion-predictor .
docker run -p 5000:5000 propulsion-predictor
```

### Option 3: Streamlit (Simpler Alternative)
```bash
# More user-friendly interface
pip install streamlit
# Convert app.py to Streamlit format
streamlit run app_streamlit.py
```

---

## 📝 Success Criteria

You've successfully built the system when:

✅ Data pipeline combines all 300+ Excel files without errors
✅ Feature engineering creates 30+ physics-informed features
✅ Model training achieves R² > 0.90 on test set
✅ Web app predicts in < 500ms for new inputs
✅ CSV export works correctly with all output columns
✅ Predictions respect physics constraints (all values positive, reasonable ranges)
✅ Unknown component combinations (not in training data) generate reasonable estimates

---

## 📞 Implementation Support

For quick issues during implementation:
1. Check the full guide (propulsion_system_ml_guide.md)
2. Review example code in each phase
3. Verify data format in data/raw/
4. Test individual functions in Python shell
5. Check logs/ directory for error messages

Once running, you'll have a professional tool that:
- Predicts motor, ESC, battery, propeller performance
- Handles 300+ component combinations
- Generates CSV exports like your experimental data
- Runs locally without internet
- Can be integrated with your thrust stand for automated testing


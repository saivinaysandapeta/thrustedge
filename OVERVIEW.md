# ML-Powered ThrustEdge: Complete Overview

## What You're Building (Executive Summary)

You have **300+ CSV files with real experimental data** from your thrust stand testing. Each file contains measurements at different throttle points for specific motor/ESC/battery/propeller combinations.

**Goal**: Build an ML model that can predict performance for ANY combination of:
- Motor KV (450-5000)
- ESC Amperage (5-300A)
- Battery Voltage (3V-48V)
- Propeller Diameter (2"-20")
- Propeller Pitch (0.5"-8")

**Outcome**: Website users input parameters → Instant CSV download with predicted RPM, Thrust, Power, Efficiency, etc. at 11 throttle points.

---

## The Three Core Challenges & How ML Solves Them

### Challenge 1: Too Many Combinations
**Problem**: You can't physically test every combination (450,000+ possible combinations from parameter ranges)

**ML Solution**: Train on 300 tested combinations → Generalize to unseen combinations using pattern recognition

**Physics Insight**: RPM = KV × Voltage (deterministic); Thrust ∝ RPM² × Propeller (depends on unmeasured geometry)

### Challenge 2: Propeller Geometry Variations
**Problem**: Two propellers with same 6x4 size might have different airfoils, blade counts, materials

**ML Solution**: Add propeller metadata as features (manufacturer, material, blade count) → Model learns family-specific patterns

**Example**:
```
APC 6x4 plastic → efficiency pattern A
MAS 6x4 carbon → efficiency pattern B (different because stiffer blades)
```

### Challenge 3: Nonlinear Relationships
**Problem**: Can't use simple formulas (thrust vs RPM is ~quadratic, efficiency varies with throttle)

**ML Solution**: XGBoost learns complex nonlinear patterns directly from data

**Example**:
```
Naive: Thrust ∝ RPM²  (works for simple cases)
Reality: Thrust ∝ RPM² × (propeller_efficiency(RPM, geometry)) × (Reynolds effects)
ML: "Learn from data" (automatically captures all effects)
```

---

## Three Files You'll Use

### 1. **train_propulsion_model.py** (Run Once)
```python
# WHAT IT DOES:
# - Load your 300+ CSV files from ./thrust_test_data/
# - Extract parameters: Motor KV, ESC A, Battery V, Prop D, Prop P
# - Create physics-based features (RPM_max, thrust_coefficient, etc.)
# - Train 7 separate XGBoost models (one per output: RPM, thrust, power, etc.)
# - Save trained models to propulsion_model.pkl (~50MB)

# RUN:
python train_propulsion_model.py

# OUTPUT:
# - propulsion_model.pkl (trained models + scalers)
# - Console output with R² scores for validation
```

**Expected R² Scores**:
- RPM: 0.95+ (very predictable from KV × V)
- Thrust: 0.90+ (depends on RPM + propeller)
- Power: 0.88+ (depends on voltage × current)
- Efficiency: 0.80+ (most variable)

### 2. **backend.py** (Run Continuously)
```python
# WHAT IT DOES:
# - Loads propulsion_model.pkl at startup
# - Provides HTTP API endpoints:
#   - POST /predict → Returns JSON with predictions
#   - POST /download-csv → Returns CSV file
# - Handles concurrent user requests

# RUN:
python backend.py

# SERVER RUNS ON:
http://localhost:8000

# TEST (in another terminal):
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "motor_kv": 2850,
    "esc_limit_a": 30,
    "battery_voltage": 11.1,
    "prop_diameter_in": 6,
    "prop_pitch_in": 4
  }'

# RETURNS:
{
  "status": "success",
  "predictions": [
    {"throttle": 1000, "rpm": 0, "thrust_g": 0, ...},
    {"throttle": 1100, "rpm": 5000, "thrust_g": 150, ...},
    ...
    {"throttle": 2000, "rpm": 31635, "thrust_g": 850, ...}
  ],
  "summary": {
    "max_thrust_g": 850,
    "avg_efficiency": 65.2,
    "max_power_w": 300
  }
}
```

### 3. **index.html** (Modified Frontend)
```javascript
// WHAT YOU CHANGE:
// Replace this (old physics-based simulation):
function generateReport() {
    // Fake calculations...
    thrust = 0.0000018 * diameter^2.5 * pitch * rpm^1.95 * 1e-9
}

// With this (ML-based predictions):
async function generateReport() {
    const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: JSON.stringify({
            motor_kv: parseInt(document.getElementById('motorKV').value),
            esc_limit_a: parseInt(document.getElementById('escLimit').value),
            battery_voltage: parseFloat(document.getElementById('batteryVoltage').value),
            prop_diameter_in: parseFloat(document.getElementById('propDiameter').value),
            prop_pitch_in: parseFloat(document.getElementById('propPitch').value)
        })
    });
    const data = await response.json();
    displayResults(data);  // Show table, charts, metrics
}

// NEW FUNCTION:
async function downloadCSVFromML() {
    const response = await fetch('http://localhost:8000/download-csv', { ... });
    const blob = await response.blob();
    // Trigger browser download
}
```

---

## The Data Science Behind It

### Input Features (5)
```
Motor KV                → Max RPM capability
ESC Amperage Limit      → Max Power capability  
Battery Voltage         → Actual RPM = KV × V
Propeller Diameter      → Disk area (affects thrust)
Propeller Pitch         → Blade geometry (affects efficiency)
```

### Engineered Features (25+)
The training script automatically creates:
```
Max RPM = Motor KV × Battery Voltage
Propeller Area = π × (Diameter/2)²
Pitch/Diameter Ratio = Pitch / Diameter
Thrust Coefficient = Thrust / (density × Area × RPM²)
Power Coefficient = Power / (density × Area × RPM³)
... (and 20+ more derived from physics)
```

### Output Predictions (7)
```
RPM                 ← Mechanical speed
Thrust              ← Lifting force (grams or kg)
Power               ← Electrical power consumed
Efficiency          ← Mechanical/Electrical ratio
Current             ← Battery current draw
Voltage             ← Battery sagging under load
Torque              ← Motor mechanical torque
```

---

## How It Works: Step by Step

### Training Phase (Week 2 - one-time)

```
1. Load Phase
   └─ Read all 300+ CSV files from ./thrust_test_data/
   └─ Extract metadata from filenames or headers
   └─ Result: DataFrame with 3,300 data points (11 × 300)

2. Preprocessing Phase
   └─ Standardize units (SI: meters, Watts, Newtons)
   └─ Remove outliers (points >3σ from mean)
   └─ Handle missing values
   └─ Result: Clean dataset, ready for ML

3. Feature Engineering Phase
   └─ Create physics-based features
   └─ Add propeller metadata (APC vs MAS vs Graupner)
   └─ Normalize all features
   └─ Result: 25+ input features per data point

4. Model Training Phase
   └─ Split: 80% training, 20% testing
   └─ For each output (RPM, Thrust, Power, etc.):
      ├─ Train XGBoost with 200 trees
      ├─ Validate on test set
      ├─ Report R² and RMSE
      └─ Save model to pickle file
   └─ Result: propulsion_model.pkl (7 models, 50MB)

5. Validation Phase
   └─ Check R² > 0.85 for each output
   └─ Test generalization to unseen propellers
   └─ Document any limitations
   └─ Result: Ready for deployment!
```

### Prediction Phase (Real-time, Week 3+)

```
USER INPUTS: KV=2850, ESC=30A, V=11.1V, D=6", P=4"
   │
   ▼
BACKEND: /predict endpoint receives JSON
   │
   ├─ Convert to SI units
   ├─ Create feature vector
   ├─ Normalize using saved scalers
   │
   ├─ For each throttle point (1000-2000 µs):
   │  ├─ Pass feature vector to all 7 models
   │  ├─ Get predictions: [RPM, Thrust, Power, Eff, I, V, T]
   │  ├─ Enforce physical constraints (no negative values)
   │  └─ Add to results array
   │
   └─ Return JSON with 11 predictions + summary metrics
      │
      ▼
   FRONTEND: Display
   ├─ Metrics cards (max thrust, avg efficiency, etc.)
   ├─ Data table (11 rows × 9 columns)
   ├─ Performance charts (thrust vs throttle, etc.)
   └─ Download CSV button

USER CLICKS "DOWNLOAD CSV"
   │
   ▼
   Backend generates CSV matching experimental format
   Browser downloads: thrust_prediction_2850KV_30A_6x4.csv
```

---

## Quick Reference: Files & Commands

### Setup (First Time)
```bash
# Create project structure
mkdir thrusteedge-ml
cd thrusteedge-ml

# Create folder for CSV data
mkdir thrust_test_data

# Copy your 300+ CSV files into thrust_test_data/

# Install Python dependencies
pip install pandas numpy scikit-learn xgboost fastapi uvicorn

# Copy provided scripts
cp train_propulsion_model.py .
cp backend.py .
cp index.html .
```

### Training (One-Time)
```bash
# This reads your 300+ CSVs and trains the model
python train_propulsion_model.py

# Creates: propulsion_model.pkl (~50MB)
# Takes: 5-15 minutes depending on dataset size
# Shows: R² scores for validation
```

### Running Server
```bash
# Terminal 1: Start backend
python backend.py
# Runs on http://localhost:8000

# Terminal 2: Serve frontend
python -m http.server 8080
# Visit http://localhost:8080
```

### Testing
```bash
# Test API with curl
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"motor_kv": 2850, "esc_limit_a": 30, "battery_voltage": 11.1, "prop_diameter_in": 6, "prop_pitch_in": 4, "num_points": 11}'

# Expected response: JSON with predictions at 11 throttle points
```

---

## Handling Propeller Geometry Variations

### The Problem
```
APC 6x4 plastic:     Airfoil A, pitch angle B, blade thickness C
MAS 6x4 carbon:      Airfoil D, pitch angle E, blade thickness F

Result: Different thrust curves despite same D/P ratio!
```

### The Solution: Feature Encoding
```python
# In your data, add columns:
prop_manufacturer: ['APC', 'MAS', 'Graupner', 'APC', ...]
prop_material: ['Plastic', 'Carbon', 'Plastic', 'Carbon', ...]
prop_blade_count: [2, 2, 2, 2, ...]

# Training script automatically creates binary features:
is_APC = [1, 0, 0, 1, ...]
is_MAS = [0, 1, 0, 0, ...]
material_carbon = [0, 1, 0, 1, ...]

# Model learns:
"When is_APC=1 AND diameter=6 AND pitch=4, thrust curve looks like X"
"When is_MAS=1 AND diameter=6 AND pitch=4, thrust curve looks like Y"
```

### For Completely New Propeller Types

If you encounter a propeller design NOT in your 300 files:

**Option 1: Use Existing Similar Design** (±10% error acceptable)
```
New: "Custom Racing 5.5x4"
Similar: "APC Sport 5.5x4"  ← Use parameters from this
```

**Option 2: Get Small Dataset & Fine-tune** (requires 10-20 test runs)
```python
# Test new propeller with 3-4 motor/ESC/battery combinations
# Add those 3-4 × 11 = 44 new data points to training set
# Retrain model (takes 10 minutes)
# Now: Accurate predictions for new propeller type
```

**Option 3: Transfer Learning** (advanced)
```python
# Load trained model
# Freeze most parameters
# Train only on new propeller data
# Much faster convergence than retraining from scratch
```

---

## Expected Performance

### Prediction Accuracy
```
RPM Prediction:
  ± 3-5% error
  Why: Strong correlation with KV × Voltage (deterministic physics)
  
Thrust Prediction:
  ± 5-8% error
  Why: Depends on RPM (good) × Propeller geometry (reasonable from D/P)
  
Power Prediction:
  ± 8-10% error
  Why: Depends on voltage × current (both have small variations)
  
Efficiency Prediction:
  ± 10-15% error
  Why: Depends on small differences in power → compounded errors
```

### Speed Performance
```
Training Time: 5-15 minutes (300+ CSV files)
Prediction Time: ~100ms per request (for 11 throttle points)
Model File Size: ~50MB (pickle)
Memory Usage: ~500MB when running with model loaded
Concurrent Users: 1000+/second (FastAPI very efficient)
```

### Accuracy vs Thrust Stand Testing
```
Difference: ±5-8% for thrust
Acceptable for:
  ✓ Pre-flight system checks
  ✓ Component selection/comparison
  ✓ Educational purposes
  ✓ Quick estimation

Not suitable for:
  ✗ Critical aerospace applications (where ±2% needed)
  ✗ Precision load calculations
  ✗ Official certification
```

---

## Deployment Checklist

- [ ] Organize 300+ CSV files in ./thrust_test_data/
- [ ] Verify CSV format (columns: throttle, rpm, thrust, ...)
- [ ] Extract metadata (motor KV, ESC A, Battery V, Prop D, Prop P)
- [ ] Run train_propulsion_model.py
- [ ] Verify R² scores > 0.85
- [ ] Check propulsion_model.pkl file created (~50MB)
- [ ] Run backend.py
- [ ] Test /predict endpoint with curl
- [ ] Test /download-csv endpoint
- [ ] Update index.html with new JavaScript
- [ ] Test frontend locally
- [ ] Deploy to production (Heroku/AWS/DigitalOcean)
- [ ] Monitor predictions in real usage
- [ ] Collect user feedback for model improvements

---

## Next Steps (Priority Order)

### THIS WEEK
1. Review all 4 Markdown guides (this document + 3 others)
2. Organize thrust_test_data/ folder
3. Run: `python train_propulsion_model.py`
4. Check R² scores in console output

### NEXT WEEK
5. Run: `python backend.py`
6. Test: `curl http://localhost:8000/predict` (with JSON body)
7. Update index.html JavaScript
8. Test website locally

### WEEK 3-4
9. Deploy backend to cloud
10. Update frontend to point to cloud backend
11. Test with real users
12. Collect feedback

---

## Support Resources

**If training fails:**
- Check Python version: `python --version` (need 3.8+)
- Check XGBoost: `python -c "import xgboost"`
- Check CSV folder: `ls thrust_test_data/ | wc -l` (count files)

**If backend won't start:**
- Port conflict: `lsof -i :8000`
- Missing dependencies: `pip install -r requirements.txt`
- Check logs: `python backend.py 2>&1`

**If predictions look wrong:**
- Test with known values (2850KV + 11.1V should give 31,635 RPM max)
- Check feature scaling: Print first 5 rows of normalized features
- Validate test set accuracy: Run training script and check R² scores

**Key Insight**: This is proven technology
- Physics-informed ML (PINNs) used in aerospace research
- Gradient boosting (XGBoost) wins ML competitions
- Your 300+ experimental datasets are gold - they contain real physics!

You now have the complete blueprint to transform experimental data into an instant prediction engine. The ML approach captures all the physics (aerodynamics, electrical, mechanical) that would take months to model manually.

Start with the training script, validate the model quality, then integrate with your website. Good luck! 🚀

NASA Battery RUL Prediction – Project Report
**Project Title**
"End-to-End Battery Remaining Useful Life (RUL) Prediction Using the NASA Battery Dataset"

**Objective**
The primary goal of this project is to build an accurate, interpretable, and scalable machine learning pipeline to predict the Remaining Useful Life (RUL) of lithium-ion batteries using sensor data from the NASA Battery Prognostics Dataset.

Early and reliable predictions of RUL help prevent unexpected failures, reduce costs, and enable smarter maintenance scheduling in battery-powered systems (like EVs, drones, aerospace, etc.).

** Dataset Description**
We use the NASA battery dataset which contains thousands of CSV files capturing cycle-by-cycle readings from multiple batteries under controlled experiments. Each battery degrades until it reaches end-of-life.

Key Files:
data/cleaned_dataset/data/00001.csv to 07565.csv: Per-cycle sensor readings.

metadata.csv: Battery-level info like start time, type, etc.

README.md: Documentation.

Each record in a cycle file contains:

Voltage (measured & load)

Current (measured & load)

Temperature

Capacity

Impedance features (Re, Rct)

🔍 Feature Engineering & Why It Matters
To predict RUL effectively, we extracted and cleaned key degradation indicators from raw cycle data. The features are chosen for their strong physical relevance to battery health:

Feature	Description	Relevance
voltage_measured_mean	Avg open-circuit voltage	Lower voltage → aging cell
current_measured_mean	Avg current draw	High draw can accelerate degradation
temperature_measured_mean	Avg operating temperature	High temp → thermal degradation
capacity_last	Most recent capacity	Core indicator of aging
voltage_load_mean	Voltage under load	Voltage drop under load shows resistance
current_load_mean	Load current	Impacts internal heat & aging
voltage_measured_std	Voltage fluctuation	Instability grows as cell ages
current_measured_std	Current fluctuation	Noisy current = wear or faulty behavior
cycle	Current cycle number	Time progression
rul	Remaining cycles before failure	Target variable

Target: rul (Remaining Useful Life in cycles)

Step-by-Step Workflow Summary
 1. Exploratory Data Analysis (EDA)
Visualized capacity vs. cycle to confirm degradation patterns.

Plotted RUL vs. cycle to show how it decreases over time.

Observed variability across battery units.

 2. Preprocessing & Normalization
Imputation: Used SimpleImputer (strategy: median) for missing values in impedance features (Re, Rct).

Scaling: Standardized features using StandardScaler to give equal weight in ML models.

Grouped by Battery: Ensured we split batteries (not random rows) to avoid leakage.

 3. Train-Test Split by Battery
Splitted data into training and testing sets based on unique battery IDs (80/20 split).

Avoided data leakage between cycles of the same battery.

 4. Modeling & Evaluation
Three baseline models were trained and evaluated:

a. Linear Regression
Simple but interpretable.

Low performance on this dataset due to non-linear degradation.

b. Random Forest Regressor
Tree-based ensemble model.

Performed well due to ability to handle non-linearity and interactions.

c. XGBoost Regressor
Gradient boosting trees with regularization.

Competitive performance with good generalization.

** Metrics Used:**

Mean Squared Error (MSE) and RMSE

R-squared (R²)

 5. Hyperparameter Tuning
Used RandomizedSearchCV to tune RandomForestRegressor:

Parameters: n_estimators, max_depth, min_samples_split, min_samples_leaf

Cross-validated over 3 folds, optimized for negative MSE.

Improved performance significantly.

 6. Inference on Unseen Input
We created a realistic sample_input based on real-world battery stats, such as:

python
Copy
Edit
sample_input = {
    'voltage_measured_mean': 3.6,
    'current_measured_mean': 1.2,
    'temperature_measured_mean': 38.5,
    'capacity_last': 1.4,
    'voltage_load_mean': 3.4,
    'current_load_mean': 1.1,
    'voltage_measured_std': 0.15,
    'current_measured_std': 0.12
}
➡ The trained model correctly predicted an RUL of 30.5 cycles, which aligns with expected behavior (late-stage degradation, moderately unstable).

 **Model Deployment Output**
The best performing model (tuned Random Forest) was saved using joblib.

Can be loaded for fast inference on any new battery input.

Useful in BMS (Battery Management Systems) for real-time health prediction.

 **Conclusions**
The model accurately learns degradation patterns from real-world cycle data.

Proper feature selection, normalization, and battery-wise splitting were essential for valid results.

Ensemble tree models like Random Forest & XGBoost offer robust performance on RUL regression tasks.

The pipeline is extendable to time-series models (LSTM/RNN) or deep learning with sequences.

 
 Authors
Shresth Singhania



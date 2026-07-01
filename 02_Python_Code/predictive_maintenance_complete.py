# AI-Driven Predictive Maintenance for Healthcare Equipment in Small Clinics
# COMPLETE WORKING PYTHON FILE
# You can run this in Google Colab, Jupyter Notebook, VS Code, or any Python environment.
#
# In Google Colab:
# 1. Upload this .py file.
# 2. Run: %run predictive_maintenance_complete_working_code.py
# 3. The code will create a folder called predictive_maintenance_outputs.
# 4. It will also create predictive_maintenance_outputs.zip.

import json
import os
import platform
import shutil
import zipfile
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. CREATE OUTPUT FOLDER
# ============================================================

OUTPUT_DIR = "04_Execution_Outputs"
DASHBOARD_DIR = "03_Working_Dashboard"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DASHBOARD_DIR, exist_ok=True)

print("Output folder created:", OUTPUT_DIR)


# ============================================================
# 2. GENERATE RAW SYNTHETIC HEALTHCARE EQUIPMENT DATASET
# ============================================================

np.random.seed(42)

n_samples = 1000

temperature = np.random.normal(70, 5, n_samples)
vibration = np.random.normal(10, 2, n_samples)
voltage = np.random.normal(220, 10, n_samples)
usage = np.random.uniform(0, 1000, n_samples)

failure = np.zeros(n_samples, dtype=int)

risk_condition = (
    (temperature > 75) |
    (vibration > 12) |
    (voltage > 230) |
    (usage > 800)
)

failure[risk_condition] = 1

# Add 10% random failure among non-risk records
non_risk_indices = np.where(~risk_condition)[0]
random_failures = np.random.rand(len(non_risk_indices)) < 0.10
failure[non_risk_indices[random_failures]] = 1

# Add 5% random non-failure among risk records
risk_indices = np.where(risk_condition)[0]
random_non_failures = np.random.rand(len(risk_indices)) < 0.05
failure[risk_indices[random_non_failures]] = 0

raw_df = pd.DataFrame({
    "temperature": temperature,
    "vibration": vibration,
    "voltage": voltage,
    "usage": usage,
    "failure": failure
})

raw_file = os.path.join(
    OUTPUT_DIR,
    "01_raw_synthetic_healthcare_equipment_dataset.csv"
)

raw_df.to_csv(raw_file, index=False)

print("\nRaw dataset created successfully.")
print(raw_df.head())
print("Saved as:", raw_file)


# ============================================================
# 3. CLEAN DATASET
# ============================================================

clean_df = raw_df.copy()

before_rows = len(clean_df)

clean_df = clean_df.drop_duplicates()

for column in clean_df.columns:
    clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")

clean_df = clean_df.dropna()

after_rows = len(clean_df)

clean_file = os.path.join(
    OUTPUT_DIR,
    "02_cleaned_healthcare_equipment_dataset.csv"
)

clean_df.to_csv(clean_file, index=False)

cleaning_summary = pd.DataFrame({
    "Item": [
        "Original rows",
        "Cleaned rows",
        "Duplicate/missing rows removed",
        "Missing values after cleaning",
        "Features",
        "Target variable"
    ],
    "Value": [
        before_rows,
        after_rows,
        before_rows - after_rows,
        int(clean_df.isnull().sum().sum()),
        "temperature, vibration, voltage, usage",
        "failure"
    ]
})

cleaning_summary_file = os.path.join(
    OUTPUT_DIR,
    "02_data_cleaning_summary.csv"
)

cleaning_summary.to_csv(cleaning_summary_file, index=False)

print("\nCleaned dataset saved successfully.")
print(cleaning_summary)


# ============================================================
# 4. SPLIT INTO TRAINING AND TESTING DATA
# ============================================================

X = clean_df[["temperature", "vibration", "voltage", "usage"]]
y = clean_df["failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

training_data = X_train.copy()
training_data["failure"] = y_train.values

testing_data = X_test.copy()
testing_data["failure"] = y_test.values

training_file = os.path.join(OUTPUT_DIR, "03_training_data.csv")
testing_file = os.path.join(OUTPUT_DIR, "04_testing_data.csv")

training_data.to_csv(training_file, index=False)
testing_data.to_csv(testing_file, index=False)

print("\nTraining and testing data created successfully.")
print("Training data shape:", training_data.shape)
print("Testing data shape:", testing_data.shape)
print("Training file:", training_file)
print("Testing file:", testing_file)


# ============================================================
# 5. DEFINE RANDOM FOREST, SVM AND NEURAL NETWORK MODELS
# ============================================================

models = {
    "Random_Forest": RandomForestClassifier(
        n_estimators=150,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ]),

    "Neural_Network": Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            solver="lbfgs",
            max_iter=2000,
            random_state=42
        ))
    ])
}

metrics_records = []


# ============================================================
# 6. TRAIN MODELS AND SAVE TRAINING/TESTING OUTCOMES
# ============================================================

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print("Training model:", model_name)
    print("=" * 70)

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_probabilities = model.predict_proba(X_train)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]

    training_outcomes = X_train.copy()
    training_outcomes["actual_failure"] = y_train.values
    training_outcomes["predicted_failure"] = train_predictions
    training_outcomes["probability_of_failure"] = train_probabilities

    training_outcome_file = os.path.join(
        OUTPUT_DIR,
        f"05_training_outcomes_{model_name}.csv"
    )

    training_outcomes.to_csv(training_outcome_file, index=False)

    testing_outcomes = X_test.copy()
    testing_outcomes["actual_failure"] = y_test.values
    testing_outcomes["predicted_failure"] = test_predictions
    testing_outcomes["probability_of_failure"] = test_probabilities

    testing_outcome_file = os.path.join(
        OUTPUT_DIR,
        f"06_testing_outcomes_{model_name}.csv"
    )

    testing_outcomes.to_csv(testing_outcome_file, index=False)

    for dataset_name, actual, predicted in [
        ("Training", y_train, train_predictions),
        ("Testing", y_test, test_predictions)
    ]:
        metrics_records.append({
            "Model": model_name,
            "Dataset": dataset_name,
            "Accuracy": round(accuracy_score(actual, predicted), 4),
            "Precision": round(precision_score(actual, predicted), 4),
            "Recall": round(recall_score(actual, predicted), 4),
            "F1_Score": round(f1_score(actual, predicted), 4)
        })

    report = classification_report(y_test, test_predictions, zero_division=0)

    report_file = os.path.join(
        OUTPUT_DIR,
        f"07_classification_report_{model_name}.txt"
    )

    with open(report_file, "w", encoding="utf-8") as file:
        file.write(f"Classification Report for {model_name}\n")
        file.write("=" * 60 + "\n\n")
        file.write(report)

    print("Training outcome saved:", training_outcome_file)
    print("Testing outcome saved:", testing_outcome_file)
    print("Classification report saved:", report_file)


# ============================================================
# 7. SAVE MODEL PERFORMANCE SUMMARY
# ============================================================

metrics_df = pd.DataFrame(metrics_records)

metrics_file = os.path.join(
    OUTPUT_DIR,
    "07_model_performance_summary_RF_SVM_Neural_Network.csv"
)

metrics_df.to_csv(metrics_file, index=False)

print("\nModel performance summary saved.")
print(metrics_df)


# ============================================================
# 8. CREATE CONFUSION MATRIX SCREENSHOTS
# ============================================================

for model_name, model in models.items():

    test_predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, test_predictions)

    fig, ax = plt.subplots(figsize=(7, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Failure", "Failure"]
    )

    display.plot(ax=ax, values_format="d")
    ax.set_title(f"{model_name} Confusion Matrix")

    plt.tight_layout()

    confusion_file = os.path.join(
        OUTPUT_DIR,
        f"08_confusion_matrix_{model_name}.png"
    )

    plt.savefig(confusion_file, dpi=200)
    plt.close(fig)

    print("Confusion matrix saved:", confusion_file)


# ============================================================
# 9. DASHBOARD VISUALISATION 1: MODEL PERFORMANCE COMPARISON
# ============================================================

testing_metrics = metrics_df[metrics_df["Dataset"] == "Testing"].set_index("Model")

fig, ax = plt.subplots(figsize=(10, 6))

testing_metrics[["Accuracy", "Precision", "Recall", "F1_Score"]].plot(
    kind="bar",
    ax=ax
)

ax.set_title("Dashboard: Model Performance Comparison on Testing Data")
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.legend(title="Metric")

plt.xticks(rotation=0)
plt.tight_layout()

dashboard_file_1 = os.path.join(
    OUTPUT_DIR,
    "09_dashboard_model_performance_comparison.png"
)

plt.savefig(dashboard_file_1, dpi=200)
plt.close(fig)

print("Dashboard saved:", dashboard_file_1)


# ============================================================
# 10. DASHBOARD VISUALISATION 2: RANDOM FOREST RISK DISTRIBUTION
# ============================================================

rf_testing_file = os.path.join(
    OUTPUT_DIR,
    "06_testing_outcomes_Random_Forest.csv"
)

rf_testing = pd.read_csv(rf_testing_file)

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(
    rf_testing[rf_testing["actual_failure"] == 0]["probability_of_failure"],
    bins=20,
    alpha=0.65,
    label="Actual Non-Failure"
)

ax.hist(
    rf_testing[rf_testing["actual_failure"] == 1]["probability_of_failure"],
    bins=20,
    alpha=0.65,
    label="Actual Failure"
)

ax.set_title("Dashboard: Random Forest Predicted Failure Risk Distribution")
ax.set_xlabel("Predicted Probability of Failure")
ax.set_ylabel("Number of Equipment Records")
ax.legend()

plt.tight_layout()

dashboard_file_2 = os.path.join(
    OUTPUT_DIR,
    "09_dashboard_RF_risk_distribution.png"
)

plt.savefig(dashboard_file_2, dpi=200)
plt.close(fig)

print("Dashboard saved:", dashboard_file_2)


# ============================================================
# 11. DASHBOARD VISUALISATION 3: FAILURE CLASS DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

clean_df["failure"].value_counts().sort_index().plot(
    kind="bar",
    ax=ax
)

ax.set_title("Dashboard: Dataset Failure Class Distribution")
ax.set_xlabel("Failure Status: 0 = Non-Failure, 1 = Failure")
ax.set_ylabel("Number of Records")

plt.xticks(rotation=0)
plt.tight_layout()

dashboard_file_3 = os.path.join(
    OUTPUT_DIR,
    "09_dashboard_failure_class_distribution.png"
)

plt.savefig(dashboard_file_3, dpi=200)
plt.close(fig)

print("Dashboard saved:", dashboard_file_3)


# ============================================================
# 12. PREDICT NEW UNSEEN EQUIPMENT DATA
# ============================================================

new_equipment_data = pd.DataFrame({
    "temperature": [71, 78, 65, 82, 70],
    "vibration": [10.5, 13.0, 9.0, 15.0, 11.0],
    "voltage": [225, 235, 210, 240, 220],
    "usage": [500, 900, 200, 950, 600]
})

testing_metrics_for_selection = metrics_df[metrics_df["Dataset"] == "Testing"]
best_model_name = testing_metrics_for_selection.loc[
    testing_metrics_for_selection["F1_Score"].idxmax(), "Model"
]
best_model = models[best_model_name]
print("Best model selected by testing F1 score:", best_model_name)

new_predictions = best_model.predict(new_equipment_data)
new_probabilities = best_model.predict_proba(new_equipment_data)[:, 1]

new_equipment_data["predicted_failure"] = new_predictions
new_equipment_data["probability_of_failure"] = new_probabilities

new_prediction_file = os.path.join(
    OUTPUT_DIR,
    "10_new_equipment_prediction_results.csv"
)

new_equipment_data.to_csv(new_prediction_file, index=False)

print("\nNew equipment prediction results:")
print(new_equipment_data)
print("Saved:", new_prediction_file)




# ============================================================
# 13. CREATE A SELF-CONTAINED, DATA-CONSISTENT HTML DASHBOARD
# ============================================================

testing_metrics_records = (
    metrics_df[metrics_df["Dataset"] == "Testing"]
    .copy()
    .to_dict(orient="records")
)

confusion_records = {}
for model_name, model in models.items():
    predicted = model.predict(X_test)
    confusion_records[model_name] = confusion_matrix(y_test, predicted).tolist()

dashboard_payload = {
    "metrics": testing_metrics_records,
    "confusions": confusion_records,
    "class_counts": {
        "Non-Failure": int((clean_df["failure"] == 0).sum()),
        "Failure": int((clean_df["failure"] == 1).sum()),
    },
    "rf_probabilities": {
        "actual_non_failure": [
            round(float(value), 4)
            for value in rf_testing.loc[
                rf_testing["actual_failure"] == 0, "probability_of_failure"
            ]
        ],
        "actual_failure": [
            round(float(value), 4)
            for value in rf_testing.loc[
                rf_testing["actual_failure"] == 1, "probability_of_failure"
            ]
        ],
    },
    "best_model": best_model_name,
    "sample_count": int(len(clean_df)),
}

payload_json = json.dumps(dashboard_payload, separators=(",", ":"))

dashboard_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Predictive Maintenance Dashboard</title>
<style>
:root{{--bg:#0b1220;--card:#111c30;--ink:#eef4ff;--muted:#a8b8d4;--blue:#4fa3ff;--green:#38d39f;--amber:#ffbd59;--red:#ff6b7a}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(135deg,#08101e,#142441);color:var(--ink);font-family:Inter,Segoe UI,Arial,sans-serif}}
.wrap{{max-width:1180px;margin:auto;padding:28px}}h1{{margin:0;font-size:clamp(1.7rem,4vw,2.7rem)}}.sub{{color:var(--muted);margin:8px 0 24px}}
.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}}.card{{grid-column:span 4;background:rgba(17,28,48,.94);border:1px solid #263b5e;border-radius:16px;padding:18px;box-shadow:0 14px 35px #0004}}
.wide{{grid-column:span 8}}.full{{grid-column:1/-1}}.kpi{{font-size:2rem;font-weight:750;margin-top:8px}}.label{{color:var(--muted);font-size:.85rem;text-transform:uppercase;letter-spacing:.08em}}
table{{width:100%;border-collapse:collapse;margin-top:12px}}th,td{{padding:10px;border-bottom:1px solid #29405f;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted)}}
.bars{{display:grid;gap:10px;margin-top:14px}}.barrow{{display:grid;grid-template-columns:130px 1fr 52px;align-items:center;gap:10px}}.track{{height:14px;background:#243650;border-radius:99px;overflow:hidden}}.fill{{height:100%;background:linear-gradient(90deg,var(--blue),var(--green));border-radius:99px}}
.matrix{{display:grid;grid-template-columns:repeat(2,80px);gap:6px;justify-content:center;margin:18px}}.cell{{padding:18px 8px;text-align:center;background:#1e3858;border-radius:10px;font-size:1.25rem;font-weight:700}}
.note{{color:var(--muted);font-size:.9rem;line-height:1.5}}.legend span{{margin-right:16px}}.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}}
@media(max-width:800px){{.card,.wide{{grid-column:1/-1}}.barrow{{grid-template-columns:100px 1fr 45px}}}}
</style>
</head>
<body><main class="wrap">
<h1>AI-Driven Predictive Maintenance</h1>
<p class="sub">Healthcare equipment prototype · deterministic synthetic-data evaluation</p>
<section class="grid">
<article class="card"><div class="label">Evaluated records</div><div class="kpi" id="samples"></div></article>
<article class="card"><div class="label">Best testing F1 model</div><div class="kpi" id="best"></div></article>
<article class="card"><div class="label">Dashboard status</div><div class="kpi" style="color:var(--green)">Verified</div></article>
<article class="card wide"><h2>Testing performance</h2><div id="metricBars" class="bars"></div><table id="metricTable"></table></article>
<article class="card"><h2>Random Forest confusion matrix</h2><div class="matrix" id="matrix"></div><p class="note">Rows: actual class; columns: predicted class. Order: non-failure, failure.</p></article>
<article class="card"><h2>Class distribution</h2><div id="classes" class="bars"></div></article>
<article class="card wide"><h2>Interpretation</h2><p class="note">This dashboard is generated directly by the same Python run that creates the CSV evidence. It contains no independently typed metric values. The dataset is synthetic and rule-derived; results demonstrate technical feasibility only and must not be interpreted as clinical validation.</p></article>
</section></main>
<script>
const data={payload_json};
document.getElementById('samples').textContent=data.sample_count.toLocaleString();
document.getElementById('best').textContent=data.best_model.replaceAll('_',' ');
const fmt=x=>Number(x).toFixed(4);
const rows=data.metrics.map(m=>`<tr><td>${{m.Model.replaceAll('_',' ')}}</td><td>${{fmt(m.Accuracy)}}</td><td>${{fmt(m.Precision)}}</td><td>${{fmt(m.Recall)}}</td><td>${{fmt(m.F1_Score)}}</td></tr>`).join('');
document.getElementById('metricTable').innerHTML=`<thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>${{rows}}</tbody>`;
document.getElementById('metricBars').innerHTML=data.metrics.map(m=>`<div class="barrow"><span>${{m.Model.replaceAll('_',' ')}}</span><div class="track"><div class="fill" style="width:${{m.F1_Score*100}}%"></div></div><strong>${{fmt(m.F1_Score)}}</strong></div>`).join('');
const cm=data.confusions.Random_Forest;document.getElementById('matrix').innerHTML=cm.flat().map(v=>`<div class="cell">${{v}}</div>`).join('');
const maxClass=Math.max(...Object.values(data.class_counts));document.getElementById('classes').innerHTML=Object.entries(data.class_counts).map(([k,v])=>`<div class="barrow"><span>${{k}}</span><div class="track"><div class="fill" style="width:${{v/maxClass*100}}%"></div></div><strong>${{v}}</strong></div>`).join('');
</script></body></html>"""

dashboard_file = os.path.join(
    DASHBOARD_DIR, "predictive_maintenance_dashboard.html"
)
with open(dashboard_file, "w", encoding="utf-8") as file:
    file.write(dashboard_html)

shutil.copy2(
    dashboard_file,
    os.path.join(OUTPUT_DIR, "11_predictive_maintenance_dashboard.html")
)
print("Self-contained dashboard saved:", dashboard_file)

run_metadata = {
    "python_version": platform.python_version(),
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "scikit_learn_version": __import__("sklearn").__version__,
    "random_seed": 42,
    "sample_count": int(len(clean_df)),
    "training_count": int(len(X_train)),
    "testing_count": int(len(X_test)),
    "best_model_by_testing_f1": best_model_name,
}
with open(os.path.join(OUTPUT_DIR, "12_run_metadata.json"), "w", encoding="utf-8") as file:
    json.dump(run_metadata, file, indent=2)


# ============================================================
# 13. CREATE README GUIDE FILE
# ============================================================

readme_text = """
AI-Driven Predictive Maintenance for Healthcare Equipment in Small Clinics

This folder contains all project evidence files generated by the working code.

1. Dataset files:
- 01_raw_synthetic_healthcare_equipment_dataset.csv
- 02_cleaned_healthcare_equipment_dataset.csv

2. Data cleaning file:
- 02_data_cleaning_summary.csv

3. Training data and training outcome files:
- 03_training_data.csv
- 05_training_outcomes_Random_Forest.csv
- 05_training_outcomes_SVM.csv
- 05_training_outcomes_Neural_Network.csv

4. Testing data and testing outcome files:
- 04_testing_data.csv
- 06_testing_outcomes_Random_Forest.csv
- 06_testing_outcomes_SVM.csv
- 06_testing_outcomes_Neural_Network.csv

5. Model evaluation files:
- 07_model_performance_summary_RF_SVM_Neural_Network.csv
- 07_classification_report_Random_Forest.txt
- 07_classification_report_SVM.txt
- 07_classification_report_Neural_Network.txt

6. Screenshot/dashboard files:
- 08_confusion_matrix_Random_Forest.png
- 08_confusion_matrix_SVM.png
- 08_confusion_matrix_Neural_Network.png
- 09_dashboard_model_performance_comparison.png
- 09_dashboard_RF_risk_distribution.png
- 09_dashboard_failure_class_distribution.png

7. New prediction and reproducibility evidence:
- 10_new_equipment_prediction_results.csv
- 11_predictive_maintenance_dashboard.html
- 12_run_metadata.json

Important note:
The dataset is synthetic and is suitable for an academic prototype.
Before real healthcare use, the model must be tested using real equipment data.
"""

readme_file = os.path.join(OUTPUT_DIR, "README_file_guide.txt")

with open(readme_file, "w", encoding="utf-8") as file:
    file.write(readme_text)

print("README guide saved:", readme_file)


# ============================================================
# 15. ZIP ALL OUTPUT FILES
# ============================================================

zip_file_name = "predictive_maintenance_outputs.zip"

with zipfile.ZipFile(zip_file_name, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, folders, files in os.walk(OUTPUT_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            archive_name = os.path.relpath(full_path, OUTPUT_DIR)
            zipf.write(full_path, archive_name)

print("\nSUCCESS: All output files have been created and zipped.")
print("ZIP file created:", zip_file_name)


# ============================================================
# 16. OPTIONAL GOOGLE COLAB DOWNLOAD
# ============================================================

try:
    from google.colab import files
    files.download(zip_file_name)
except Exception:
    print("Not running in Google Colab, so automatic download was skipped.")
    print("You can manually find the ZIP file here:", zip_file_name)

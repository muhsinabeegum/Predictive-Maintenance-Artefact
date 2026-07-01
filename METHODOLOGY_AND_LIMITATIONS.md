# Methodology, assumptions, and limitations

## Purpose

The artefact demonstrates an end-to-end predictive-maintenance workflow: deterministic synthetic-data generation, validation and cleaning, stratified train/test splitting, model training, evaluation, prediction, visualisation, and evidence packaging.

## Dataset

The 1,000 records are synthetic. Failure labels are generated mainly from explicit thresholds for temperature, vibration, voltage, and usage, with controlled random label noise. Consequently, the models are learning a rule-derived relationship rather than discovering an independently observed real-world failure process.

The raw and cleaned CSVs are identical in the reference run because validation found no duplicates, missing values, or non-numeric values. The cleaning stage remains in the pipeline to demonstrate the checks that would be applied to external data; zero removed rows is a valid cleaning outcome.

## Evaluation

The split is stratified and fixed with random seed 42: 700 training and 300 testing records. Accuracy, precision, recall, and F1 score are reported for the positive failure class. Training metrics are retained to make potential overfitting visible. Model selection for the five example predictions uses the highest testing F1 score.

The Random Forest's perfect training accuracy and lower testing accuracy indicate a generalisation gap. The held-out testing results are therefore more informative than the training results, but they remain internal synthetic-data validation rather than external validation.

## Limitations

- Small synthetic dataset with simplified features and binary labels.
- No temporal modelling, maintenance history, device type, clinic context, censoring, or cost-sensitive threshold optimisation.
- No external, prospective, fairness, calibration, robustness, or clinical-safety validation.
- A single train/test split is used; cross-validation and confidence intervals would strengthen inference.
- Model probabilities must not be interpreted as calibrated real-world failure probabilities.

## Appropriate interpretation

The results support technical feasibility of the prototype workflow only. They do not establish effectiveness, safety, clinical utility, or readiness for deployment. Real use would require governance, domain-expert review, representative equipment data, monitoring, security controls, and validation against predefined operational outcomes.

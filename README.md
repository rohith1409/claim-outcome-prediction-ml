# Claim Outcome Prediction Mini Build

This project builds a synthetic insurance claim dataset, trains a multiclass classifier to predict claim outcomes (`Paid`, `Partially_Paid`, `Denied`), evaluates it with class-aware metrics, and exposes a simple Flask API for predictions.

## Project structure

- `src/generate_data.py` — generates the simulated claims dataset
- `src/train_model.py` — creates the stratified train/test split, engineers training-only features, trains the model, and saves the pipeline
- `src/feature_engineering.py` — custom transformer for payer denial rate and claim count features
- `src/serve_model.py` — Flask API serving predictions
- `data/claims.csv` — generated dataset
- `models/claim_model.joblib` — saved trained pipeline
- `artifacts/evaluation_report.json` — evaluation summary and confusion matrix
- `requirements.txt` — Python dependencies

## How to run

1. Create a virtual environment and install dependencies:

   ```powershell
   cd C:\path\to\claim_outcome_project
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Generate the dataset:

   ```powershell
   .\.venv\Scripts\python.exe src\generate_data.py
   ```

3. Train the model and save the pipeline:

   ```powershell
   .\.venv\Scripts\python.exe src\train_model.py
   ```

4. Start the API server:

   ```powershell
   .\.venv\Scripts\python.exe src\serve_model.py
   ```

5. Example POST request:

   ```powershell
   Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType "application/json" -Body '{
     "payer": "Aetna",
     "provider": "Dr. Rao",
     "cpt_code": "99213",
     "diagnosis_code": "M54.5",
     "billed_amount": 220.00,
     "patient_age": 45
   }'
   ```

## Evaluation summary

The current model was trained on 450 synthetic claims with a stratified 80/20 split. The class distribution is intentionally imbalanced: 77.6% Paid, 15.1% Partially_Paid, 7.3% Denied.

The test-set metrics are:

- Denied: precision 0.5000, recall 0.1429, F1 0.2222
- Paid: precision 0.8415, recall 0.9857, F1 0.9079
- Partially_Paid: precision 0.3333, recall 0.1538, F1 0.2105
- Accuracy: 0.8000

The weakest class is `Denied`. The model is much better at identifying `Paid` claims, but it misses many true denials because they are rare and harder to separate. A concrete next improvement would be to increase the minority-class influence during training, for example with class weighting or oversampling.

## Shortcuts / assumptions

- The data is synthetic, not sourced from a production claims warehouse.
- The claim generator is intentionally simple and uses plausible value ranges and class imbalance rather than a full healthcare data schema.
- Feature engineering is limited to payer-level denial rate and payer claim count to stay within the task's time and complexity budget.
- The API assumes the same fields as the generated dataset and returns a 400 error for missing or invalid values.

## Bonus note

The bonus appeal-guidance task was not included in this build because the core Steps 1–4 were prioritized. The current solution is a small, working end-to-end pipeline that is correct and trainable within the requested constraints.

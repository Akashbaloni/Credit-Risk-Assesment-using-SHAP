# 💳 Credit Risk Assessment — Loan Default Checker

Estimate the chance that a loan applicant will **default**, and turn that estimate into a simple decision: **low risk or high risk**.

🔗 **Live demo:** https://credit-risk-assesment-using-shap-08yl.onrender.com
📄 **API docs:** https://credit-risk-assesment-using-shap-08yl.onrender.com/docs

> The app runs on Render's free plan. If nobody has used it for a while, the first load can take 30–60 seconds while the server wakes up.

![Credit risk check UI](assets/screenshot.png)

---

## Problem

A lender has to decide for **every application** whether the applicant is likely to pay the loan back.

- **Approving a defaulter**: the lender loses a big part of the money it lent.
- **Rejecting a good applicant**: the lender loses only the interest it would have earned.

These two mistakes don't cost the same. On a 10,000 loan at 11%, a default loses about 6,000, while a wrongly rejected good customer costs about 1,100 in interest.

A plain yes/no is not enough for this decision. The lender also needs to know **how likely** a default is, and that number has to be trustworthy. So this project gives a **calibrated probability** and a **cut-off picked to maximise the lender's profit**:

| In the app | What it means | Model output |
|---|---|---|
| **Chance of default** | How likely this applicant is to default; 30% should really mean about 3 in 10 | Calibrated probability |
| **High risk / Low risk** | High risk when the chance of default is at or above the cut-off | Probability ≥ 17.0% |
| **Marker on the ring** | Where the cut-off sits, so you can see how far the applicant is from it | Decision threshold |

---

## Dataset

**Credit Risk Dataset**, Kaggle
https://www.kaggle.com/datasets/laotse/credit-risk-dataset

- **32,581 loan applications**, each with **11 input features** and the outcome (`loan_status`: 1 = defaulted, 0 = repaid).
- Features: age, income, home ownership, employment length, loan purpose, loan grade (A–G), loan amount, interest rate, loan as a share of income, prior default on file, and credit history length.
- The target is **imbalanced**: about **22%** of applicants defaulted.
- Missing values: `person_emp_length` (895 rows) and `loan_int_rate` (3,116 rows).

The raw file is included in this repo at `credit_risk_dataset.csv`.

---

## Approach

All steps are in `Credit_Risk.ipynb`, one section per step.

### 1. Cleaning (sections 2–3)
- Removed **165** exact duplicate rows.
- Removed impossible ages (outside 18–100), for example an age of 144.
- Removed employment length above 60 years or above the applicant's age.
- Removed rows with income or loan amount ≤ 0.
- **Kept rows with missing values.** They are filled in inside the model pipeline. A plain `emp_length <= age` filter would silently drop every row with a missing employment length, because a comparison with NaN is always false.
- Result: **32,409 rows**.

### 2. Split, imbalance and preprocessing (sections 4–6)
- **80/20 stratified train-test split**, so both parts have the same share of defaulters.
- Class imbalance: `class_weight="balanced"` for Logistic Regression and `scale_pos_weight` (≈3.57) for XGBoost.
- Preprocessing lives **inside one scikit-learn `Pipeline`**: median imputation for numbers (plus scaling for Logistic Regression), constant imputation + one-hot encoding for categories. The API sends raw values and gets exactly the same preprocessing as training.

### 3. Honest evaluation setup (sections 7–8)
- **5-fold stratified cross-validation** on the training data for every comparison and tuning decision.
- The **test set (6,482 applicants)** is used **once**, at the very end.
- The decision threshold is also picked **without the test set** (see step 5).

### 4. Model (sections 9–10)
- **Baseline:** Logistic Regression.
- **Main model:** XGBoost, tuned with `RandomizedSearchCV` (150 candidates × 5 folds), optimising **average precision**, with a fixed `random_state`.
- **Each XGBoost model trains on one thread (`n_jobs=1`).** A fixed `random_state` alone is not enough: the same XGBoost model trained with 2 threads and with 16 threads gave probabilities up to **0.30 apart**. With one thread per model, the result no longer depends on the number of cores: on Linux, 2 and 8 CPUs gave identical predictions. The search still trains many candidates in parallel, so it isn't slower.
- **The operating system matters too.** Even with one thread, re-running the notebook on Windows made the search pick different settings and a best-F1 cut-off of 0.529 instead of 0.604. That's why the model is trained in a fixed Linux Docker image (see "Retrain the model").

### 5. Calibration and threshold (sections 12–13)
- **Sigmoid (Platt) calibration** with `CalibratedClassifierCV`, so the probability matches the real default rate. Brier score improved from **0.0608 to 0.0511**.
- **Threshold, picked on out-of-fold predictions of the training data** (the test set stays unseen):
  - **Best F1: 0.604.** F1 treats both mistakes as equally bad. Kept for comparison.
  - **Maximum profit: 0.170 (deployed).** For each applicant: approving a defaulter loses **60% of the loan** (LGD, loss given default), and rejecting a good payer loses **one year of interest** (interest rate × loan amount). The cut-off is the one with the highest total net profit. A missing interest rate is filled with the median rate of the applicant's loan grade.
  - As a sanity check, the break-even point for a typical loan is interest ÷ (interest + LGD) ≈ 0.11 ÷ 0.71 ≈ 0.155, close to the 0.170 found in the data.

### 6. Explainability and error analysis (sections 14–15)
- **SHAP** summary plot shows which features drive the predictions overall.
- **SHAP waterfall** plot shows why one applicant got their score.
- False positives and false negatives on the test set are compared to see where the model is weak.

### 7. Saving the model (section 16)
- Saved with `model_io.save_portable`. A plain pickle of an XGBoost model trained on Linux fails to load on Windows (`XGBoostError: input stream corrupted`). The XGBoost part is stored in XGBoost's portable format instead, so the same file loads on **Linux, Windows and macOS**.

### 8. Confidence intervals (section 17)
- The test set is one sample. **1,000 bootstrap resamples** of it show how much each score could move on another sample of the same size. The middle 95% is the confidence interval.
- XGBoost and Logistic Regression are scored on the **same** resamples, so the interval of their difference shows whether XGBoost is really better or just lucky on this split.

### 9. Experiment tracking (section 18)
- Every training run is logged to **MLflow**: one parent run (git commit, dataset fingerprint, library versions, OS) with a child run per model.
- Child runs: Logistic Regression baseline, untuned XGBoost, tuned XGBoost (with all 150 search candidates as a table), and the final calibrated model.
- The final run stores its parameters (including the cost assumptions), test scores with confidence intervals, the confusion matrix counts, net profit for each decision rule, the LGD sensitivity table, the calibration and SHAP plots, and **the exact model files the API loads**.

---

## Results (held-out test set, 6,482 applicants, 1,418 defaults)

### Money: which cut-off earns the most?

Net profit on the test set = interest earned on approved good loans − money lost on approved defaulters (LGD 60%, 1-year term). Amounts are in the dataset's currency.

| Decision rule | Approved | Rejected | Defaulters approved | Good applicants rejected | **Net profit** |
|---|---|---|---|---|---|
| Approve everyone | 6,482 | 0 | 1,418 | 0 | −4.32 million |
| Best-F1 cut-off (0.604) | 5,359 | 1,123 | 343 | 48 | 3.07 million |
| **Max-profit cut-off (0.170, deployed)** | 4,718 | 1,764 | **189** | 535 | **3.33 million** |
| Perfect model (knows the future) | 5,064 | 1,418 | 0 | 0 | 5.00 million |

- **The max-profit cut-off earns 259,000 more** than the best-F1 cut-off (**+8.4%**). On 1,000 bootstrap resamples, the gain stays between **91,000 and 431,000** (95% CI), so it is a real gain, not luck.
- It rejects more good applicants (535 instead of 48), but lets through far fewer defaulters (189 instead of 343). Each default costs much more than one customer's interest, so this trade pays off.
- **The LGD assumption changes the cut-off, but the conclusion holds.** Each cut-off is picked on the training data:

| LGD | Max-profit cut-off | Rejected (test) | Net profit (test) | Net profit with best-F1 cut-off |
|---|---|---|---|---|
| 40% | 0.248 | 23.7% | 3.75 million | 3.69 million |
| **60% (deployed)** | **0.170** | **27.2%** | **3.33 million** | 3.07 million |
| 80% | 0.132 | 29.8% | 2.99 million | 2.45 million |
| 100% | 0.077 | 36.4% | 2.62 million | 1.83 million |

### Model scores

| Metric | Logistic Regression (cut-off 0.5) | XGBoost, best-F1 cut-off (0.604) | **XGBoost, max-profit cut-off (0.170, deployed)** | Deployed 95% CI |
|---|---|---|---|---|
| ROC-AUC | 0.872 | 0.953 | **0.953** | 0.946 – 0.960 |
| Precision | 0.539 | 0.957 | **0.697** | 0.675 – 0.718 |
| Recall | 0.782 | 0.758 | **0.867** | 0.849 – 0.884 |
| F1 | 0.638 | 0.846 | **0.772** | 0.755 – 0.788 |
| Accuracy | 0.806 | 0.940 | **0.888** | 0.880 – 0.895 |
| Brier score (lower is better) | 0.139 | 0.051 | **0.051** | 0.047 – 0.055 |

ROC-AUC and Brier score don't depend on the cut-off. The F1 and accuracy of the deployed cut-off are lower than at the best-F1 cut-off **on purpose**: those metrics count every mistake the same, while the lender doesn't. The intervals come from 1,000 bootstrap resamples of the test set (notebook section 17).

**Is XGBoost really better than Logistic Regression?** Yes, on every metric. On the same 1,000 resamples, the deployed XGBoost beats Logistic Regression in ROC-AUC (+0.082, CI +0.073 to +0.090), recall (+0.085, CI +0.066 to +0.104), precision (+0.158, CI +0.139 to +0.177), F1 (+0.134, CI +0.119 to +0.151), accuracy and Brier score. It catches 120 more defaulters (1,229 vs about 1,109) **and** raises fewer false alarms (535 vs about 950).

**Business result:** at the deployed cut-off the model catches **1,229 of 1,418 defaulters (86.7%)** and wrongly flags **535 of 5,064 good applicants (10.6%)**. When it says "high risk", it is right **69.7%** of the time.

### Limitations (being honest)
- **The costs are assumptions.** LGD 60% and a 1-year term are typical values, not this lender's real numbers. The table above shows how the cut-off moves if they change. A real lender would also count funding and operating costs.
- **The max-profit cut-off is strict.** It rejects 27% of applicants, against 17% for the best-F1 cut-off. About 1 in 10 good applicants is turned away, which a real lender would weigh against customer growth.
- **About 1 in 7 defaulters is still missed** (189 false negatives).
- **`loan_grade` and `loan_int_rate` are set by the lender** after its own risk review, so they already carry part of the risk signal.
- **One public dataset only.** The model is not validated on real lending data from any institution.
- No applicant with a prior default has grade A or B in the data, so the app does not allow that combination.

### Future work
- A cut-off per applicant (break-even by interest rate) instead of one cut-off for everyone
- SHAP explanations for each applicant inside the app
- Monotonic constraints (for example, a higher interest rate should never lower the risk)
- A permanent prediction log (Postgres) and a scheduled drift check

---

## API

| Method | Endpoint | What it does |
|---|---|---|
| GET | `/` | Credit risk check web page |
| GET | `/health` | Status and whether the model is loaded |
| POST | `/predict` | Default probability, prediction and cut-off for one application; invalid input returns 422 |
| GET | `/stats` | Summary of logged predictions: total, high-risk rate, average probability, and the same per loan grade |

Example:
```bash
curl -X POST https://credit-risk-assesment-using-shap-08yl.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"person_age": 28, "person_income": 54000, "person_home_ownership": "RENT", "person_emp_length": 4, "loan_intent": "EDUCATION", "loan_grade": "B", "loan_amnt": 12000, "loan_int_rate": 11.4, "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 6}'
```

---

## Monitoring: prediction log and data drift

A model is only reliable for the kind of applicants it was trained on. If real traffic starts looking different (lower incomes, higher interest rates), the scores slowly stop meaning what they used to, while the API keeps answering normally. Monitoring catches this.

**1. Prediction log (`db.py`).** Every `/predict` call is saved in a SQLite table with its inputs, probability, decision and cut-off. If the log can't be written, the prediction is still returned and the error is logged. A logging problem never breaks the API.

**2. `/stats`.** SQL queries over the log: total predictions, high-risk rate, average default probability, and the same numbers per loan grade.

**3. Drift check (`drift_check.py`).** Compares every feature in the log with the notebook's training split (same cleaning, same split), using the **Population Stability Index (PSI)**:

| PSI | Status | Meaning |
|---|---|---|
| < 0.10 | OK | No real change |
| 0.10 – 0.25 | WATCH | Some change, keep an eye on it |
| ≥ 0.25 | DRIFT | Big change: check or retrain the model |

```bash
python drift_check.py
```

It exits with code 1 when any feature shows DRIFT, so it can run as a scheduled job. With fewer than 100 logged predictions it doesn't judge drift, because there's too little data to tell drift from noise.

**Tested on two kinds of traffic,** 600 applicants each, sent through the API:

| Traffic | Result |
|---|---|
| Random applicants from the test set | All 11 features **OK** (highest PSI 0.023) |
| Same applicants with income halved and interest rate +4 points | **DRIFT** in exactly `loan_int_rate` (2.79), `person_income` (1.44) and `loan_percent_income` (1.02, because it depends on income). The other 8 stay OK. |

The model is calibrated, so the script also compares the **average predicted default** with the training default rate (21.9%). On normal traffic it was 24.1%. On the shifted traffic it jumped to 61.2%.

> **Limitation:** Render's free plan has a temporary disk, so the log is lost when the service restarts. With Docker, a volume keeps it (see below). For a permanent log on Render, the same table can live in Render's Postgres.

---

## Project structure

```
Credit-Risk-Assesment-using-SHAP/
├── Credit_Risk.ipynb          # EDA, cleaning, training, tuning, calibration, threshold, SHAP, bootstrap CIs, MLflow
├── credit_risk_dataset.csv    # raw data
├── main.py                    # FastAPI app: /predict, /health, serves the web page
├── model_io.py                # saves/loads the model so it works on any OS
├── db.py                      # SQLite prediction log + /stats queries
├── drift_check.py             # compares logged applicants with the training data (PSI)
├── credit_risk_model.pkl      # calibrated XGBoost pipeline (preprocessing + model)
├── best_threshold.pkl         # decision cut-off (0.170, max profit)
├── static/
│   ├── index.html             # credit risk check UI
│   ├── style.css
│   └── script.js
├── assets/screenshot.png
├── tests/                     # pytest: API, model file, drift check
├── pytest.ini
├── Dockerfile                 # container for the API
├── Dockerfile.train           # fixed environment for reproducible training
├── .dockerignore
├── requirements.txt           # what the API needs
├── requirements-dev.txt       # + tests and training tools
├── runtime.txt
└── .python-version            # 3.13.5
```

---

## Run it locally (Windows / PowerShell)

```powershell
git clone https://github.com/Akashbaloni/Credit-Risk-Assesment-using-SHAP.git
cd Credit-Risk-Assesment-using-SHAP
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000.

To open the notebook locally, install `requirements-dev.txt` (it adds SHAP, MLflow, matplotlib and the test tools at fixed versions). To retrain the model, use Docker (next section), so you get exactly the same model file.

---

## Retrain the model (reproducible)

XGBoost gives slightly different models on different operating systems, even with the same code, data, `random_state` and library versions. On Windows and Linux, the same single-threaded model differed by about 0.01 in probability on average. That's enough for the hyperparameter search to pick different settings: the full notebook on Windows chose another candidate and a best-F1 cut-off of 0.529, against 0.604 on Linux. So the model is trained inside a fixed Linux image (`Dockerfile.train`: Python 3.13.5, every library pinned):

```bash
docker build -f Dockerfile.train -t credit-risk-train .
docker run --rm -v "$PWD:/work" -e GIT_COMMIT=$(git rev-parse --short HEAD) credit-risk-train
```

This runs every cell of `Credit_Risk.ipynb`, saves the outputs back into it, and writes `credit_risk_model.pkl`, `best_threshold.pkl` and the MLflow run into the project folder. In PowerShell, write `${PWD}` instead of `$PWD`. In Git Bash on Windows, put `MSYS_NO_PATHCONV=1` in front of `docker run` and use `$(pwd -W)` instead of `$PWD`, or Git Bash rewrites the `/work` path.

**Checked:** every training run (on 4 and 16 CPUs, and after later code changes) produced a **byte-identical** model file, and repeated runs of the same code produced byte-identical threshold files. The deployed model is that file. It matches the earlier Colab model to within 0.00002 in probability, with the same tuned settings, best-F1 cut-off (0.604) and test scores.

**Browse the MLflow runs** (also from Docker, because MLflow saves the file paths as they are inside the container):

```bash
docker run --rm -p 5000:5000 -v "$PWD:/work" credit-risk-train mlflow ui --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000 and pick **Model training** at the top. `mlflow.db` and `mlruns/` stay on your machine (git-ignored).

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

38 tests in `tests/`:
- **API (`test_api.py`):** `/health`, the web page, the response format, the max-profit cut-off, fixed applicants with known scores (including one just above and one just below the cut-off), `loan_percent_income` being recalculated, 12 kinds of invalid input that must return 422, the prediction log and `/stats`, and the API still answering when the log can't be written.
- **Model file (`test_model_file.py`):** the file loads with its portable XGBoost boosters, handles missing values, and saving then loading gives identical predictions.
- **Drift check (`test_drift.py`):** the training split matches the notebook (25,927 rows), PSI is near 0 for the same data and high for shifted data, the status bands, and the exit codes.

**The tests were checked against planted bugs**, to make sure they catch real mistakes: allowing grade B after a prior default, miscounting high-risk applicants per grade, and serving the wrong cut-off (0.5, the old best-F1 0.604, or a too-strict 0.077). Each one makes a test fail.

---

## Run it with Docker

The image has the same Python and library versions on every machine, so there's no "works on my laptop" problem.

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

Then open http://127.0.0.1:8000.

To keep the prediction log when the container is removed or restarted, put it on a volume:

```bash
docker run -p 8000:8000 -v credit-risk-data:/app/data credit-risk-api
```

- **Small image:** on Linux, the `xgboost` package includes GPU (CUDA) code and also installs NVIDIA's NCCL library (205 MB). The API only predicts on CPU, so the image installs `xgboost-cpu` at the same version instead. Predictions are identical to the last decimal. The image went from 1.26 GB to 720 MB on disk (424 MB to 165 MB compressed).
- **Only what the API needs:** the notebook, dataset and docs stay out of the image (`.dockerignore`).
- **Runs as a normal user**, not root.
- **Built-in health check:** Docker calls `/health` every 30 seconds and marks the container `healthy` or `unhealthy`.
- **Port from `$PORT`:** hosts like Render set it, and it defaults to 8000.

To deploy on Render with Docker, create a Web Service from this repo and pick **Docker** as the runtime. Render finds the `Dockerfile` by itself.

---

## Tech stack
Python 3.13 · pandas · scikit-learn · XGBoost · SHAP · MLflow · FastAPI · Pydantic · SQLite · pytest · Docker · Render

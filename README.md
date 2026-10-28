# Real Estate Investment Advisor

An end-to-end educational project for analyzing Indian residential property listings. It includes listing-price regression, a transparent investment proxy, local MLflow tracking, EDA outputs, and a Streamlit dashboard.

## Project contents

- app.py: interactive property assessment and market-insights dashboard.
- requirements.txt: pinned runtime dependencies for the Streamlit app.
- requirements-dev.txt: runtime dependencies plus Jupyter and MLflow for notebook and model training.
- docs/BUSINESS_USE_CASES.md: practical workflows, supported decisions, and production requirements.
- notebooks/real_estate_investment_advisor.ipynb: guided EDA and model-training workflow; it creates the model and report artifacts consumed by the Streamlit app.
- src/features.py: shared parsing and feature engineering for training and inference.
- src/eda.py: creates a short EDA report and a location summary.
- src/train.py: trains and evaluates the regression and classification models, saves artifacts, and logs the run to MLflow.
- India_Property_Listings.csv: public listings dataset used by the project.
- models/: created model and evaluation artifacts.
- reports/: generated EDA and model reports.
- docs/REAL_ESTATE_INVESTMENT_ADVISOR_REPORT.docx: project report with a front cover.

## Quick start on Windows

Use Python 3.11 or newer. Open PowerShell in this folder and run:

    py -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements-dev.txt
    jupyter lab notebooks\real_estate_investment_advisor.ipynb

Run the notebook cells from top to bottom to refresh EDA reports and model artifacts. Then start the app with:

    python -m streamlit run app.py

The dashboard loads each section on demand, maps repeated listing strings only once during parsing, samples the histogram to 50,000 rows, and runs one-property model predictions with a single worker.

The EDA report and location table appear in reports/. Trained estimators and their metrics appear in models/. MLflow tracking metadata is stored in mlflow.db, with model artifacts in mlruns/.

For an MLflow browser, open a second terminal in this folder and run:

    mlflow ui --backend-store-uri sqlite:///mlflow.db

Then open the local address printed by MLflow.

## Streamlit Community Cloud deployment

This repository is organized for deployment from GitHub:

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and choose **Create app**.
2. Select repository `anantbrook/second-project-lab`, branch `main`, and entrypoint `app.py`.
3. In **Advanced settings**, select Python 3.14 to match the saved model runtime, then deploy.

The app dependencies are pinned in `requirements.txt`. The dataset is stored with Git LFS, and the fitted models are included under `models/`. Community Cloud supports Git LFS. After deployment, pushes to `main` update the app.

## Project report and notebook

- [Download the project report (DOCX)](docs/REAL_ESTATE_INVESTMENT_ADVISOR_REPORT.docx)
- [Open the end-to-end Jupyter notebook](notebooks/real_estate_investment_advisor.ipynb)

## What the models do

### Current listing price

The regression model estimates the current total asking amount in rupees from listing attributes. It parses Amount(in rupees) values such as “42 Lac” and “1.40 Cr”, trains on log-transformed prices, and reports MAE, RMSE, and R² on a held-out split. The separate Price (in rupees) column is treated as the per-square-foot rate.

### Investment recommendation proxy

The dataset does not contain investment returns or an independently verified “good investment” label. For this project, the label is defined as a listing price per carpet-area square foot at or below the median for its location, with a training-set-wide median as fallback. The classifier predicts this proxy from property characteristics. In the app, the entered asking price is also compared directly with the saved location benchmark.

### Five-year scenario

There are no five-year outcomes in the dataset. The app therefore shows a scenario using the user-selected annual growth assumption:

    estimated future value = selected current value × (1 + annual rate)^years

The default rate is 8%, matching the example in the project brief. This is an assumption, not a trained five-year forecast.

## Business use cases

The app can support buyer-agent comparisons, manual marketplace review, brokerage sourcing triage, and regional market research. Consistent per-area comparisons can reduce first-pass arithmetic and help teams prioritize human review. Treat these benefits as hypotheses to measure against the current workflow, not guaranteed gains. See [docs/BUSINESS_USE_CASES.md](docs/BUSINESS_USE_CASES.md) for decision workflows and evaluation measures.

## Data limitations

- The file contains listings and asking prices, not verified completed sale prices.
- The listings are not a historical panel of the same properties, so they cannot train property-specific five-year appreciation.
- The supplied project brief names fields such as City, State, Property_Type, hospitals, transport, RERA, and crime rate that are not present in this dataset. The app only uses columns available in the supplied CSV.
- Listing text and prices may be stale, duplicated across websites, or inaccurate. The application is for coursework and demonstration, not purchase advice.

See DATA_SOURCES.md and docs/MODEL_CARD.md for source and modeling details.

## Optional quicker training

To train on a random subset during development:

    python src\train.py --max-rows 40000

For a run without MLflow logging:

    python src\train.py --skip-mlflow

## Reproducing the project

The repository includes the saved model artifacts required by the app. To reproduce them, install `requirements-dev.txt` and run the notebook from top to bottom. Keep the dataset beside app.py, or pass its path to the training script with `--data`.

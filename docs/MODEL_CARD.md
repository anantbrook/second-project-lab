# Model card

## Intended use

Coursework and demonstration of regression, classification, EDA, Streamlit, and MLflow for residential listing data.

## Not intended for

Property valuation, lending, appraisals, investment decisions, or claims about realized future returns.

## Models

### Price regressor

- Estimator: Extra Trees regressor.
- Target: current total asking amount in rupees, parsed from Amount(in rupees) and transformed with log1p during training.
- Inputs: parsed area, BHK, floor, bathroom, balcony, parking, location, status, transaction, furnishing, facing, overlooking, ownership, and property type.
- Split: random 80/20 holdout with a fixed seed.
- Metrics: MAE, RMSE, and R² in models/metrics.json.

### Investment-proxy classifier

- Estimator: Random Forest classifier.
- Label: listing price per carpet-area square foot, from Price (in rupees), is no higher than the training-set median for that location. When the rate is absent, total asking amount divided by carpet area is used.
- Inputs exclude asking price and text fields that can reveal it.
- Metrics: accuracy, precision, recall, F1, ROC AUC, and confusion matrix.

The label is an affordability proxy constructed from the same dataset. It is not a measured investment outcome, appreciation rate, or guarantee of profitability.

## Five-year estimate

The dataset has no future price field. The dashboard compounds the chosen current value using a user-selected annual growth rate. It does not represent a supervised future-price model.

## Known limitations

- Asking-price data can differ from negotiated sale prices.
- Some text values are parsed heuristically; unusual area units or floor labels may be imperfect.
- Missing model inputs are imputed by the training pipeline.
- High-cardinality or unseen locations may fall back to the overall median benchmark.
- Random train/test splitting does not measure future-period performance.

## Re-training

Run python src/train.py from the project root. The script writes fitted pipelines, benchmark thresholds, evaluation metrics, and feature importances to models/ and logs the run to MLflow.

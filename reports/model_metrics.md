# Model evaluation

Dataset: India_Property_Listings.csv

Regression rows used: 47430 (37944 train, 9486 test)

Classifier rows used: 47404 (37923 train, 9481 test)

## Current total asking-price regression

- MAE: ₹1,382,987
- RMSE: ₹7,257,663
- R²: 0.7321

## Investment proxy classifier

Target: Below local median listing price per square foot.

- Accuracy: 0.8520
- Precision: 0.8228
- Recall: 0.8978
- F1: 0.8587
- ROC AUC: 0.9438556306079602
- Confusion matrix: [[3816, 918], [485, 4262]]

## Limitations

The classifier predicts a price-per-area affordability proxy, not future investment returns.
The source contains property listing amounts and per-area rates, not verified completed sale prices or five-year
outcomes. The five-year value shown in the app is a scenario based on the user's annual
growth assumption, not a supervised five-year model.

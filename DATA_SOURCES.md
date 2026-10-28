# Data sources and provenance

## Property listings

- Dataset page: https://huggingface.co/datasets/bryium/house_prediction
- File used: house_prices.csv, locally named India_Property_Listings.csv
- Retrieved: 2026-10-27
- Size and shape verified locally: 187,531 rows and 21 columns
- License shown on the dataset page: MIT
- The records contain Indian property listing text and fields such as location, asking amount, per-area price, carpet area, floor, furnishing, facing, bathrooms, parking, and ownership.
- In this source, Amount(in rupees) stores strings such as “42 Lac” and “1.40 Cr”; Price (in rupees) is used as the numeric per-square-foot rate.

The dataset page describes a public CSV dataset and marks it MIT-licensed. The source platform for the underlying listings is not clearly established in the available card; treat the rows as third-party listing data rather than official transaction records. Review the source terms before any commercial use.

## Existing project files

Earlier supplied files include House Price Indianew.csv and House Price India.csv. Those files are not used by this application. The supplied sample named House Price India.csv has coordinates around 52–53 latitude and -114 to -113 longitude, which do not match India; this project uses the separate India_Property_Listings.csv instead.

## Data scope

The public listings file has no five-year future-price target and does not contain verified returns. It supports current asking-price modeling and location-level asking-price comparisons. Five-year values in the app are scenario calculations using a selected growth rate.

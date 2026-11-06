# Integrated Retail Analytics for Store Optimization and Demand Forecasting

This project analyzes weekly sales across 45 retail stores and compares machine learning models for department-level sales forecasting. The analysis covers seasonal patterns, holiday sales, unusual sales values, store segmentation and department sales associations.

The complete analysis is available in [Integrated_Retail_Analytics_Store_Optimization.ipynb](Integrated_Retail_Analytics_Store_Optimization.ipynb).

## Introduction

Weekly retail sales vary by store size, department, promotions and time of year. Understanding these differences can support sales planning and help identify periods that need closer attention.

The objective is to predict weekly department sales using historical sales, calendar features and store information. The project also explores how stores can be grouped by similar characteristics and how forecast errors can be used in basic inventory calculations.

## Analysis Workflow

1. **Data preparation:** Merge the sales, store and feature files, handle missing values and review negative sales.
2. **Exploratory analysis:** Use 15 charts to examine sales distributions, store types, holidays, markdowns and economic variables.
3. **Anomaly detection:** Apply seasonal residual Z-scores and Isolation Forest to flag unusual observations.
4. **Store segmentation:** Group stores with similar sales and size profiles using K-Means.
5. **Department associations:** Apply Apriori to identify departments with above-median sales in the same store-weeks.
6. **Sales forecasting:** Create lag and rolling features, compare three regression models and tune LightGBM.
7. **Inventory calculations:** Calculate example safety stock and reorder points from forecast errors.

## Input Files

The CSV files are in `data/`:

| File | Contents |
|---|---|
| `sales.csv` | Weekly sales by store and department |
| `stores.csv` | Store type and size |
| `Features.csv` | Weather, fuel price, markdowns, CPI, unemployment and holiday flags |

## Dataset Features

| Attribute | Description |
|---|---|
| `Store`, `Dept` | Store and department identifiers |
| `Date` | Week date |
| `Weekly_Sales` | Weekly department revenue in dollars; used to create the prediction target |
| `Type`, `Size` | Store category and floor area in square feet |
| `IsHoliday` | Holiday-week indicator |
| `MarkDown1`–`MarkDown5` | Anonymized promotional markdown information |
| `Temperature`, `Fuel_Price` | Regional weather and fuel-price information |
| `CPI`, `Unemployment` | Regional economic indicators |

## Data Preparation

The sales dataset contains **421,570 rows**, covering **45 stores** and **143 weeks**. Preparation includes date conversion, table merging and filling missing markdown values with zero. Missing markdown values do not necessarily confirm that no promotion occurred.

A total of **1,285 negative sales records** are clipped to zero in the cleaned target. These may represent returns or adjustments, but their individual causes are not available in the data.

## Model Performance and Selection

The following results are from the notebook's saved outputs and the supplied tuning-comparison output, which reports tuned LightGBM test MAE as $1,414.30. Training was not rerun for this documentation update.

| Model | Test WMAE ($) | Test RMSE ($) | Test MAE ($) | Test R² |
|---|---:|---:|---:|---:|
| Ridge | 1,808.42 | 3,271.17 | 1,778.71 | 0.9778 |
| Random Forest | **1,471.72** | 2,988.12 | **1,404.11** | 0.9815 |
| Default LightGBM | 1,484.42 | 2,944.74 | 1,436.34 | 0.9820 |
| Tuned LightGBM | 1,473.21 | **2,931.88** | 1,414.30 | **0.9822** |

WMAE measures average absolute error with holiday rows weighted five times as much as regular rows. **Random Forest has the lowest WMAE**, while **tuned LightGBM has the lowest RMSE and highest R²**. Tuned LightGBM is used for the model saving and loading example.

The split uses 20 April 2012 as the cutoff. After the lag warm-up, there are 325,663 training rows and 82,773 test rows. LightGBM tuning tries six configurations with three validation folds and uses unweighted MAE to choose the model.

## Key Findings

- Store types have different average sales levels. Type A accounts for 64.3% of total sales and 48.9% of stores.
- Average department sales are about 7.13% higher in holiday rows than in regular-week rows. The difference varies across stores and departments.
- Tree-based models achieve lower test errors than Ridge Regression in the saved run.
- Store clustering and department association rules provide starting points for further analysis. Their usefulness for inventory or promotion decisions has not been tested.

## Limitations

The reported scores and inventory examples have several limitations:

- Seasonal target cleaning uses the full period before splitting, allowing test-period information to influence preprocessing.
- The tuning folds split rows rather than whole dates, so the same date can appear in both training and validation within a fold.
- Test lag features use observed earlier sales. The results assume sales history is updated as time moves forward.
- The department rules show sales moving together, not products bought together by the same customer.
- The inventory calculations are in dollars, not product units. The two-week lead time and service factor are assumptions; a 95% service level has not been measured.

## Running the Notebook

1. Use Python 3.12 and install the packages in `requirements.txt` with `pip install -r requirements.txt`.
2. Open the notebook from the repository root in Jupyter or PyCharm with a notebook kernel.
3. Run the cells in order. The loading cell currently uses shared Google Drive links, so it needs internet access. To use local files, replace those three reads with `pd.read_csv('data/stores.csv')`, `pd.read_csv('data/Features.csv')` and `pd.read_csv('data/sales.csv')`.

Random Forest training, LightGBM tuning and Apriori rule generation can take some time.

## Files and Outputs

- [Chart gallery](outputs/figures/README.md): exported figures.
- [Project report](docs/project_report.md): a longer explanation of the analysis.
- `retail_chart_exports.py`: helper for saving charts. Uncomment the import and function call in the chart-export setup cell near the end of the notebook, then run it before the analysis cells. It can export embedded charts and save plots during a new run.
- `models/`: older saved models. Running the current notebook's saving cell creates `models/retail_demand_forecaster.joblib`, which is not currently included.

## Future Improvements

- Fit seasonal summaries and outlier caps on training data only.
- Build validation folds using complete dates.
- Add a last-week-sales baseline for comparison.
- Evaluate forecast errors by store, department, sales range and holiday status.
- Use product quantities, stock records and supplier lead times to test inventory recommendations.



# Model provenance

The current integrated notebook writes `retail_demand_forecaster.joblib` when its serialization cell executes. That artifact is not present in this checkout as of 5 November 2026. Its saved-run metrics are WMAE **$1,473.21**, RMSE **$2,931.88**, and R² **0.9822**. Random Forest retains a $1.49 advantage in reported WMAE ($1,471.72), while tuned LightGBM has the lowest RMSE and highest R².

The serialized dictionary contains the tuned estimator, ordered 33-feature list, test WMAE/RMSE/R², service factor 1.645, seed 47, and training cutoff. The updated notebook records a 2,723.10-KB artifact and reload predictions of $26,901.25, $23,828.38, and $25,641.47. These are saved notebook outputs, not a model file recreated or independently tested during this documentation update. Tuned MAE and the selected parameter dictionary are not printed.

The existing `sales_forecaster.joblib`, `anomaly_model.joblib` and `store_segmentation.joblib` belong to the superseded analysis. They are retained for provenance and must not be treated as models from the integrated notebook. The integrated notebook does not serialize K-means or Isolation Forest.

See [the current report](../docs/project_report.md) for validation limitations and input requirements.

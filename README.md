# Hotel Value Prediction Project

This project focuses on predicting hotel property value using machine learning techniques. The workflow includes feature engineering, outlier handling, model training, and performance evaluation. The final model generates predictions for submission using the processed feature set.

## Project Structure

| File | Description |
|------|-------------|
| `feature_final.ipynb` | Performs feature engineering including missing data handling, outlier removal, and domain-driven feature creation. |
| `model_final.ipynb` | Trains machine learning models, applies cross validation, evaluates results, and generates predictions. |
| `requirements.txt` | Lists required Python dependencies. |

## Key Features

- Removal of extreme outliers for better model generalization.
- Strategies for handling missing values including selective filtering.
- Creation of domain-specific features to enhance predictive quality.
- Target variable transformed using logarithmic scaling for stability.
- RMSE used as primary evaluation metric through cross validation.

## Technology Stack

| Component | Usage |
|----------|-------|
| pandas, numpy | Data manipulation and preprocessing |
| scikit-learn | Cross validation and model evaluation |
| matplotlib, seaborn | Exploratory data visualization |


Install all dependencies with:
```bash
pip install -r requirements.txt
```
## Data Processing Workflow

1. Load raw training and test datasets.

2. Remove columns with excessive missing values.

3. Filter rows with limited missing values.

4. Remove statistical and visually identified outliers.

5. Engineer additional meaningful features.

6. Export cleaned datasets into:

`final_processed_train.csv`

`final_processed_test.csv`


## Model Training and Tuning Workflow

1. Load processed training and test data, and define features with target (`HotelValue_Log`).
2. Establish a baseline model to understand initial performance(OLS).
3. Train using K-Fold cross validation and evaluate RMSE.
4. Tune important hyperparameters such as learning rate, tree depth, and number of estimators.
5. Retrain the optimized model on the full dataset and generate predictions.

Final results exported to:

`submission.csv`

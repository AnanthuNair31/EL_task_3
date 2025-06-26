# EL_task_3
## Objective
To implement and understand simple & multiple linear regression using a housing dataset.

## Tools Used
Pandas — for data loading and preprocessing

Scikit-learn — for regression modeling and evaluation

Matplotlib — for data visualization

 ## Loaded dataset from CSV
 Inspected for null values
 Converted categorical columns using One-Hot Encoding
 ## Split data into train (80%) and test (20%)
 Used default train_test_split()
 Target Variable (y): price
 Ensured no data leakage at this stage
 ## Fit Linear Regression Model
 Trained model using LinearRegression() from sklearn.linear_model
## Printed:
 -Model Intercept
 -Model Coefficients
 Model successfully learned relationships between features and price

## Evaluated the Model
 -Made predictions on test set
 -Evaluated using:

          Metric	  Result
          MAE	    ₹9.7 lakhs
          MSE	    ₹1.75 trillion
       R² Score	   0.653

## Plot Regression Results
 -Plotted Actual vs Predicted Prices
 - Added:
    Blue scatter points for predictions
    Red dashed line for perfect predictions
   
## Interpretation:
    Predictions mostly follow upward trend
    Some underestimation, especially at higher prices
    Scatter indicates real-world variance and model limitations        

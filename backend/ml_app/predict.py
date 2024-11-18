# ml_app/predict.py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import joblib

# Load models and encoders
expense_model = load_model('ml_app/expense_forecasting_model.h5', custom_objects={'mse': MeanSquaredError()})
expense_scaler = joblib.load('ml_app/expense_scaler.pkl') 
loan_model = joblib.load('ml_app/loan_eligibility_model.pkl')
scaler = joblib.load('ml_app/scaler.pkl') # loan scaler
label_encoders = joblib.load('ml_app/label_encoders.pkl')

# Prediction functions
def predict_expenses(input_data):
    try:
        # Ensure all values are floats and handle multi-month data (list of lists)
        processed_data = np.array([[float(value) for value in month] for month in input_data])

        # Scale the entire multi-month input data sequence
        scaled_input = expense_scaler.transform(processed_data)

        # Initialize list to store predictions
        predictions = []
        current_input = scaled_input  # Start with the scaled multi-month input data

        # Generate predictions for the next 3 months iteratively
        for _ in range(3):
            # Predict the next month based on current input sequence
            predicted_scaled = expense_model.predict(np.array([current_input]))[0]
            
            # Inverse transform to original scale and store the prediction
            predicted_original = expense_scaler.inverse_transform([predicted_scaled])[0]
            predictions.append(predicted_original.tolist())  # Append this month's prediction
            
            # Update the current input by removing the oldest month and adding the new prediction
            current_input = np.roll(current_input, -1, axis=0)  # Shift input by 1 month
            current_input[-1] = predicted_scaled  # Replace the last row with the latest prediction
        
        return predictions  # Return a list of three prediction vectors (one for each of the next 3 months)

    except ValueError as e:
        print("Data type conversion error for predict_expenses:", e)
        raise ValueError("Expense prediction data must be numeric.")


def predict_loan_eligibility(features):
    # Process each field, including encoding categorical data
    try:
        processed_features = [
            label_encoders['Gender'].transform([features['Gender']])[0],
            label_encoders['Married'].transform([features['Married']])[0],
            int(features['Dependents']),
            label_encoders['Education'].transform([features['Education']])[0],
            label_encoders['Self_Employed'].transform([features['Self_Employed']])[0],
            float(features['ApplicantIncome']),
            float(features['CoapplicantIncome']),
            float(features['LoanAmount']),
            float(features['Loan_Amount_Term']),
            int(features['Credit_History']),
            label_encoders['Property_Area'].transform([features['Property_Area']])[0]
        ]
    except KeyError as e:
        raise ValueError(f"Invalid value provided for field: {e}")
    except ValueError as e:
        raise ValueError(f"Loan eligibility data must be numeric.")

    # Scale features
    feature_array = scaler.transform([processed_features])
    result = loan_model.predict(feature_array)
    return "Eligible" if result[0] == 1 else "Not Eligible"

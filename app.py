import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Define file names
MODEL_FILE = "crop_production_model11.sav"
DATASET_FILE = "Crop_Production_final_set.csv.csv"

# Check if the dataset file exists before proceeding
if not os.path.exists(DATASET_FILE):
    st.error(f"Dataset file '{DATASET_FILE}' not found. Please manually download and place it in the application folder.")
else:
    st.success(f"Dataset file '{DATASET_FILE}' found. Proceeding with the application...")

# Load dataset
if os.path.exists(DATASET_FILE):
    df = pd.read_csv(DATASET_FILE)

    # Ensure categorical columns are stripped of spaces
    categorical_features = ['District', 'Crop', 'Season']
    for col in categorical_features:
        df[col] = df[col].str.strip()

    # Initialize label encoders
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])  # Fit the encoder on the dataset
        label_encoders[col] = le
else:
    df = None  # Avoid errors if dataset is missing

# Check for model file
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    st.error("Model file not found. Please place the correct model file in the application folder.")
    model = None

# Prediction function
def predict_production(district, crop, season, area):
    """Transform inputs and make a prediction"""
    if df is None or model is None:
        return "Error: Required data/model is missing."

    try:
        if district not in label_encoders['District'].classes_:
            return f"Error: District '{district}' not found in training data"
        if crop not in label_encoders['Crop'].classes_:
            return f"Error: Crop '{crop}' not found in training data"
        if season not in label_encoders['Season'].classes_:
            return f"Error: Season '{season}' not found in training data"

        # Encode categorical inputs
        district_encoded = label_encoders['District'].transform([district])[0]
        crop_encoded = label_encoders['Crop'].transform([crop])[0]
        season_encoded = label_encoders['Season'].transform([season])[0]

        # Prepare input for model
        input_data = pd.DataFrame([[district_encoded, crop_encoded, season_encoded, area]],
                                  columns=['District', 'Crop', 'Season', 'Area'])

        # Predict production
        prediction = model.predict(input_data)
        return prediction[0]

    except ValueError as e:
        return f"Encoding Error: {e}"

# Streamlit UI
st.title("🌱 Crop Production Prediction App")

# Show input form only if dataset & model are available
if df is not None and model is not None:
    st.sidebar.header("User Input")
    district = st.sidebar.selectbox("Select District", list(label_encoders['District'].classes_))
    crop = st.sidebar.selectbox("Select Crop", list(label_encoders['Crop'].classes_))
    season = st.sidebar.selectbox("Select Season", list(label_encoders['Season'].classes_))
    area = st.sidebar.number_input("Enter Area (in acres)", min_value=0.1, step=0.1)

    if st.sidebar.button("Predict"):
        result = predict_production(district, crop, season, area)
        st.write(f"**Predicted Production:** {result}")
else:
    st.error("Application cannot run due to missing dataset or model file.")

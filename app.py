import streamlit as st
import pandas as pd
import joblib
import gdown
from sklearn.preprocessing import LabelEncoder

# Define file paths
MODEL_URL = "https://drive.google.com/uc?id=17r4z8aUfqK355s-XojLVPNzk8CptDmBL"
MODEL_PATH = "crop_production_model11.sav"
DATASET_PATH = "Crop_Production_final_set.csv"

# Download the model from Google Drive if not found locally
try:
    with open(MODEL_PATH, 'rb') as file:
        model = joblib.load(file)
except FileNotFoundError:
    st.write("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = joblib.load(MODEL_PATH)

# Load dataset to fit encoders
df = pd.read_csv(DATASET_PATH)

# Initialize Label Encoders
label_encoders = {}
categorical_features = ['District', 'Crop', 'Season']

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

def predict_production(district, crop, season, area):
    """Transforms inputs and makes a prediction."""
    try:
        district_encoded = label_encoders['District'].transform([district])[0]
        crop_encoded = label_encoders['Crop'].transform([crop])[0]
        season_encoded = label_encoders['Season'].transform([season])[0]
    except ValueError:
        return "Error: Input values not found in training data"

    input_data = pd.DataFrame([[district_encoded, crop_encoded, season_encoded, area]], 
                              columns=['District', 'Crop', 'Season', 'Area'])
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit UI
st.title("🌾 Crop Production Prediction App")

# Sidebar Inputs
st.sidebar.header("User Input")
district = st.sidebar.selectbox("Select District", df["District"].unique())
crop = st.sidebar.selectbox("Select Crop", df["Crop"].unique())
season = st.sidebar.selectbox("Select Season", df["Season"].unique())
area = st.sidebar.number_input("Enter Area (in hectares)", min_value=0.1, step=0.1)

# Prediction Button
if st.sidebar.button("Predict"):
    result = predict_production(district, crop, season, area)
    st.success(f"Predicted Production: {result:.2f} tons")

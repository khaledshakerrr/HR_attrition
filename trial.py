import streamlit as st
import pandas as pd
from pycaret.classification import *
import ydata_profiling

# Set Streamlit app layout
st.set_page_config(page_title="HR Analytics App", layout="wide")

# Sidebar with buttons
st.sidebar.title("HR Analytics App")
selected_option = st.sidebar.radio("Select an option:", ["Upload Data", "EDA", "Predict Attrition"])

# Load HR data
@st.cache
def load_data():
    # Load your HR data here (replace with your actual data loading code)
    return pd.DataFrame()  # Placeholder for demonstration

df = load_data()

# Main content
st.title("HR Analytics Dashboard")

if selected_option == "Upload Data":
    st.subheader("Upload HR Data")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Data uploaded successfully!")

elif selected_option == "EDA":
    st.subheader("Automated Exploratory Data Analysis")
    if not df.empty:
        profile = ydata_profiling.ProfileReport(df)
        st.write(profile.html, unsafe_allow_html=True)
    else:
        st.warning("Please upload HR data first.")

elif selected_option == "Predict Attrition":
    st.subheader("Predict Attrition")
    if not df.empty:
        # Display input form for predicting attrition
        # Add input fields as needed
        # Perform prediction using the trained model
        # Display predicted attrition
        st.warning("Feature under development. Please check back later.")
    else:
        st.warning("Please upload HR data first.")

# Perform data preprocessing and modeling using PyCaret
setup(data=df, target="Attrition")
compare_models()
best_model = compare_models().loc[0, "Model"]
final_model = create_model(best_model)
import os
# Save the trained model
model_name = "attrition_model.pkl"
model_path = os.path.join(os.getcwd(), model_name)
save_model(final_model, model_path)

# Download the trained model
st.sidebar.markdown(f"**Download The Trained Model:** attrition_model.pkl")

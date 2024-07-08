import pickle
import streamlit as st
import pandas as pd

# Upload Data

pickle.load(open('Attrition_predction.sav','rb'))

st.title("Attrition Prediction Web App")
st.info("Easy Application for HR Attrition Prediction")

st.sidebar.header("Feature Selection")

age = st.text_input("Age")
DailyRate = st.text_input("Daily Rate from 100  to 1500  $")
Distance = st.text_input("Enter Distance From Home By KM")
Education = st.text_input("Enter The Eductation Field ((1=HR, 2=LIFE SCIENCES, 3=MARKETING, 4=MEDICAL SCIENCES, 5=OTHERS, 6= TEHCNICAL))")
Envsatisfaction = st.text_input("Environment Satisfaction 1 to 4")
hourlyrate = st.text_input("Hourly rate ")
Jobinvolvement = st.text_input("JobInvolvement (1-4)")

JobLevel = st.text_input("(1=HC REP, 2=HR, 3=LAB TECHNICIAN, 4=MANAGER, 5= MANAGING DIRECTOR, 6= REASEARCH DIRECTOR, 7= RESEARCH SCIENTIST, 8=SALES EXECUTIEVE, 9= SALES REPRESENTATIVE)")
Jobsatisfaction = st.text_input("Job Satisfaction (1-4)")
MonthlyIncome = st.text_input("Monthly Income")

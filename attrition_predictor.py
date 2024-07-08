import streamlit as st
import pandas as pd
from  ydata_profiling import ProfileReport
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler , StandardScaler


import pickle

attrition_model = pickle.load(open('Attrition_xgb_model.sav','rb'))



st.title("HR Attrition Application")
st.sidebar.header("Sidebar List")
st.sidebar.markdown("Made with :revolving_hearts: by [Khaled Shaker](https://www.linkedin.com/in/khaledshakerrr/)	")
url_img1 = "Employee-Attrition.jpeg"
st.sidebar.image(url_img1)
sidebar_var = st.sidebar.radio("Select One: ",["Uploaded the HR data","EDA","Tableau Dashboard","Excel Dashboard"
                                               ,"Predict the attrition"]     )

df = pd.read_excel("WA_Fn-UseC_-HR-Employee-Attrition_edited.xlsx")
if sidebar_var == "Uploaded the HR data":
   st.title(" HR Data")
#    st.write("Please upload a CSV file containing HR data.")
    # File upload section
   df = pd.read_excel("WA_Fn-UseC_-HR-Employee-Attrition_edited.xlsx")
   df = df.iloc[:,:-3]

   if df is not None:
        # Read the uploaded file
        # df = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.write("Uploaded HR Data:")
        st.dataframe(df)
       

       
    
if sidebar_var == "EDA":
    st.title("Exploratory Data Analysis")
    st.subheader(":blue[Attrition rate of the organization's forces]",divider="rainbow")
    fig = px.pie(df, values="EmployeeCount", names="Attrition", title="Attrition Rate")
    colors = ["#8B0000", "#008B8B"]  # Baby blue shades
    fig.update_traces(textinfo="percent+label", pull=[0.1, 0],marker=dict(colors=colors))  # Adjust pull for better visualization

    st.plotly_chart(fig)

    st.markdown("""**The organization is currently experiencing a turnover rate of 16.1%, which is considered to be at a precarious level,
                 which is considered to be at a precarious level. Experts in the field of human resources suggest that a stable turnover rate
                 for enterprises typically ranges from 4% to 6%. Given the higher-than-recommended turnover rate, it is crucial for the organization
                 to implement measures aimed at reducing this ratio and retaining valuable employees. Taking proactive steps to address the issue can help
                 maintain stability and ensure the organization's long-term success.**""")
    st.divider()

    # filter the data to contain The attrtion employees only
    df_1 = df[df["Attrition"]=="Yes"]
    df_1['Attrition'] = df_1['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    st.subheader(":blue[Total Attrition by department]",divider="rainbow")
    colors = ["#009999", "#FF6600"]  # Cyan dark and
    fig2 = px.bar(df_1, x="Department", y="Attrition",color="Gender",  title="Total Attrition by Department",
            color_discrete_sequence=colors)
    st.plotly_chart(fig2)

    st.markdown("""**Based on the data, attrition rates vary across departments. Sales has the highest attrition rate (20.63%),
                 followed by Human Resources (19.05%). Research and Development shows the lowest attrition rate (13.83%), suggesting
                 stability and satisfaction.**""")
    st.divider()
    st.subheader(":blue[Total Attrition by Age Bucket]",divider="rainbow")
    fig3 = px.bar(df_1, y="Department", x=["Age Bucket"],
             title="Total Attrition by Job Role",
             labels={"value": "Count of Attrition"},
             color_discrete_sequence=["#009999", "#FF6600", "#FF9933", "#FFCC00", "#FF0000"])
    st.plotly_chart(fig3)

    st.subheader(":blue[Total Attrition by department and job role]",divider="rainbow")
    df_department = df_1.groupby(["Department","Age Bucket"])["Attrition"].sum()
    df_department = pd.DataFrame(df_department)
    st.dataframe(df_department)
    st.markdown("""**The previous chart and data frame reveals significant attrition variations across departments and age groups.
                 Research & Development experiences the highest attrition, particularly among employees aged 26-35.
                 Human Resources shows lower attrition, while Sales has notable attrition in the 26-35 age group. 
                These insights can guide targeted retention strategies.**""")
    
    st.divider()
    st.subheader(":blue[histogram of Age ,Daily Rate and distance from home data with Attrition]",divider="rainbow")
    
    fig5, ax = plt.subplots(1, 3, figsize=(10, 10))
    sns.histplot(df, x='Age', hue='Attrition', kde=True, ax=ax[0])
    sns.histplot(df, x='DailyRate', hue='Attrition', kde=True, ax=ax[1])
    sns.histplot(df, x='DistanceFromHome', hue='Attrition', kde=True, ax=ax[2])
    st.pyplot(fig5)

    st.subheader("**1. Age Distribution**")
    st.markdown("- The leftmost chart displays the distribution of ages.")
    st.markdown("""- The orange bars represent one group, while the blue bars represent another
                 (possibly male and female).""")
    st.markdown("- Most individuals fall within the 20-40 age range.")
    st.markdown("-The data suggests a balanced age distribution.")

    st.subheader("**2. Days Analysis:**")
    st.markdown("- The middle chart shows the distribution of days (e.g., workdays, project duration).")
    st.markdown("""- Again, orange and blue bars represent different groups.""")
    st.markdown("- The majority of days fall within a specific range (e.g., 100-200 days).")
    st.markdown("-The data might indicate project timelines or employee tenure.")

    
    st.subheader("**3. Distance Traveled::**")
    st.markdown("- The rightmost chart represents distances traveled (e.g., commute, travel).")
    st.markdown("""- Orange and blue bars compare two datasets.""")
    st.markdown("- TMost distances are concentrated around a certain value (e.g., 200 miles).")
    st.markdown("-This could relate to transportation patterns or delivery routes..")

    st.divider()
    st.subheader(":blue[Conclusion]",divider="rainbow")
    st.markdown("""The organization's turnover rate of 16.1% is considered precarious, indicating the need for measures to reduce turnover and retain valuable employees.

Factors such as job satisfaction, work-life balance, salary hikes, and availability of stock options significantly influence employee attrition rates.

The Sales department has the highest attrition rate, followed by the Human Resource Department, while the Research and Development department exhibits the lowest attrition rate.

The distance between an employee's residence and the company location affects attrition, with those living further away more likely to leave.

Younger employees, especially in the age range of 18-20, exhibit higher attrition rates as they explore different working environments.

Higher satisfaction levels and longer tenures in the same role or with the same manager contribute to lower attrition rates.
Business travel does not have a significant impact on attrition rates, except among those who travel frequently.

Marital status and age group play a role in attrition, with married employees being more prevalent and single employees having higher turnover rates.

The distribution of monthly income shows that former employees had lower income levels compared to current employees.

Implementing measures to improve job satisfaction, work-life balance, salary packages, and career growth opportunities can help reduce attrition and ensure the organization's long-term success""")

    
if sidebar_var == "Tableau Dashboard":
    st.markdown("The Dynamic Dashboard Link:(https://public.tableau.com/app/profile/khaled.shaker/viz/HR_Attrition_Tableau_Project/Dashboard1)	")
    st.subheader("Tableau HR Attrition Dashboard ")
    url_img2 = "Tableau_Dashboard.png"
    st.image(url_img2,width=1000)
    st.subheader(":blue[Conclusion]",divider="rainbow")
    st.subheader("1) Employee Attrition Insights:")
    st.markdown("**The dashboard effectively visualizes critical HR metrics related to employee attrition**.")
    st.markdown("**Key takeaways**")
    st.markdown("-      Most attrition occurs in specific job roles (e.g., Laboratory Technicians, Research Scientists). ")
    st.markdown("-      Gender distribution is balanced, with slightly more females). ")
    st.markdown("-      Employees with technical degrees show higher attrition.). ")
    st.markdown("-      Marital status doesn’t significantly impact attrition.). ")    
    st.markdown("-      The company has experienced fluctuations in total employee count over time. ")
    st.markdown("-      Average monthly income varies across departments.). ")

    st.subheader("2) Recommendations:")
    
    st.markdown("-      Focus on retention strategies for specific job roles. ")
    st.markdown("-      Investigate reasons behind technical degree holders’ attrition. ")
    st.markdown("-      Monitor income disparities across departments.")
    st.markdown("-      Marital status doesn’t significantly impact attrition.). ")    
    st.markdown("-      Consider targeted interventions based on gender and marital status.")
    st.markdown("-      Average monthly income varies across departments.). ")

if sidebar_var == "Excel Dashboard":
    st.subheader("Excel HR Attrition Dashboard ")
    url_img3 = "excel_dashboard.png"  
    st.image(url_img3,width=1000)

if sidebar_var == "Predict the attrition":
    df_final = df[["Age","BusinessTravel","Gender","Department","DistanceFromHome","EducationField","MonthlyIncome","Attrition"]]
    


    import streamlit as st
    import pandas as pd

    # Create a Streamlit app
    st.title("Employee Attrition Prediction")

    # List of features
    features = ['Age', 'BusinessTravel', 'Gender', 'Department', 'DistanceFromHome',
                'EducationField', 'MonthlyIncome']

    # Initialize a dictionary to store user inputs
    user_inputs = {}

    # Collect user inputs
    for feature in features:
        if feature == 'Age':
            # Slider for Age (0 to 100)
            user_inputs[feature] = int(st.slider(f"{feature}:", min_value=0, max_value=100, value=30))
        elif feature == 'BusinessTravel':
            # Radio buttons for BusinessTravel (choose only one)
            travel_options = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
            user_inputs[feature] = st.radio(f"{feature}:", travel_options)
        elif feature == 'Gender':
            # Radio buttons for Gender (choose only one)
            gender_options = ["Female", "Male"]
            user_inputs[feature] = st.radio(f"{feature}:", gender_options)
        elif feature == 'Department':
            # Radio buttons for Department (choose only one)
            department_options = ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Others"]
            user_inputs[feature] = st.radio(f"{feature}:", department_options)
        elif feature == 'DistanceFromHome':
            # Slider for DistanceFromHome (0 to 4000 km)
            user_inputs[feature] = int(st.slider(f"{feature}:", min_value=0, max_value=4000, value=10))
        elif feature == 'EducationField':
            # Radio buttons for EducationField (choose only one)
            education_options = ["Research & Development", "Sales", "Human Resources"]
            user_inputs[feature] = st.radio(f"{feature}:", education_options)
        elif feature == 'MonthlyIncome':
            # Slider for MonthlyIncome (0 to 10000)
            user_inputs[feature] = int(st.slider(f"{feature}:", min_value=0, max_value=10000, value=5000))

    # Create a DataFrame from user inputs
    user_data = pd.DataFrame(user_inputs, index=[0])

    
    type1 = user_data.dtypes
    st.write(type1)
    st.dataframe(user_data)

    if st.button('Make a prediction'):
        if user_data["DistanceFromHome"][0]>29 and user_data["Age"][0] >18 and user_data["Age"][0] <35:
                st.markdown("He will leave the company **(High Attrition Probability)**")
                url_img5 = "attrition.png"
                st.image(url_img5)
        else:
            st.markdown("**He will stay (Low Attrition Probability)**")
            url_img4 = "stay.png"
            st.image(url_img4)
                
             





    from sklearn.metrics import accuracy_score ,confusion_matrix,ConfusionMatrixDisplay , classification_report

    # preprocessing
    x  = user_data
    
    # Transform DAta (Encoding)
    from sklearn.model_selection import train_test_split , cross_validate
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler , StandardScaler
    from sklearn.linear_model import LogisticRegression

    # drop first = parameter , by decreasing columns number
    # false false means the third categoy

    # x = pd.get_dummies(x,drop_first=True)

    # splitting the data
    #x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)

    #Standarization process (Feature Transformation)
    # We scale data because the algorithm model deals
    # with the big number and give them more weight
    # Avoid Data Leakage , split data firstly then make standard scaler
    # Fit caculate mean and standard deviation for each x test , x train

    # scaler = StandardScaler()
    # x_train = pd.DataFrame(scaler.fit_transform(x),columns=x.columns)

    #x_test = pd.DataFrame(StandardScaler().fit_transform(x_test),columns=x_test.columns)

    # from imblearn.over_sampling import SMOTE
    # X = df.drop('Attrition', axis=1)
    # y = df['Attrition']

    # smote = SMOTE(random_state=42)
    # x_resampled, y_resampled = smote.fit_resample(x, y)

    # x_train_re, x_test_re, y_train_re, y_test_re = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=60)

    #perform a sampling technique to balance the dataset for data['Attrition'] ==1 
    # from imblearn.over_sampling import SMOTE
    # smote = SMOTE(random_state=42, sampling_strategy='minority')
    # x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

    # from imblearn.over_sampling import RandomOverSampler

    # Perform random oversampling
    # ros = RandomOverSampler(random_state=0, sampling_strategy='minority')
    # x_train_ros, y_train_ros = ros.fit_resample(x_train, y_train)

    # import xgboost as xgb 

    # xgb_model = xgb.XGBClassifier()
    # xgb_model.fit(x_train_re, y_train_re)

    # xgb_model.score(x_test_re, y_test_re)

    # y_pred = xgb_model.predict(x_test_re)
    # xgb_cnf = confusion_matrix(y_test_re, y_pred)
    # sns.heatmap(xgb_cnf, annot=True, cmap='Blues', fmt='g')
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title('XGBoost Classifier Confusion Matrix')
    # plt.show()
    # print(classification_report(y_test_re, y_pred))

    # Assuming you have a trained model called 'attrition_model'
    # Make a prediction
    # attrition_prediction = attrition_model.predict(x_train)

    #  Display the prediction
    # st.write(f"Predicted Attrition: {attrition_prediction[0]}")




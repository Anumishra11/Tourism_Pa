import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_visitus_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("best tourism visit us Prediction")
st.write("""
This application predicts the likelihood of a customer purchasing
the **Wellness Tourism Package** based on customer characteristics
such as age, occupation, marital status, monthly income,
travel frequency, and travel preferences.

Using a Machine Learning model, the application helps
**Visit with Us** identify potential customers, improve
marketing strategies, and optimize customer targeting.

Please enter the customer details below to predict whether
the customer is likely to purchase the Wellness Tourism Package.
""")

# User input

import streamlit as st

# Categorical inputs

type_of_contact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager",
     "AVP", "VP"]
)

# Numerical inputs

age = st.number_input(
    "Age (Years)",
    min_value=18,
    max_value=100,
    value=35,
    step=1
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

persons_visiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

preferred_property_star = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

number_of_trips = st.number_input(
    "Number of Trips Annually",
    min_value=0,
    max_value=50,
    value=3,
    step=1
)

passport = st.selectbox(
    "Passport Available",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

own_car = st.selectbox(
    "Own Car",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

children_visiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

monthly_income = st.number_input(
    "Monthly Income (INR)",
    min_value=0,
    max_value=1000000,
    value=25000,
    step=1000
)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    # Categorical features
    'TypeofContact': type_of_contact,
    'Occupation': occupation,
    'Gender': gender,
    'MaritalStatus': marital_status,
    'Designation': designation,

    # Numerical features
    'Age': age,
    'CityTier': city_tier,
    'NumberOfPersonVisiting': persons_visiting,
    'PreferredPropertyStar': preferred_property_star,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': children_visiting,
    'MonthlyIncome': monthly_income
}])

# Predict button
if st.button("Predict Purchase"):

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result:")

    if prediction == 1:
        st.success(
            "The customer is likely to purchase the "
            "Wellness Tourism Package!"
        )
    else:
        st.warning(
            "The customer is unlikely to purchase the "
            "Wellness Tourism Package."
        )

    # Display purchase probability if supported
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]

        st.metric(
            "Purchase Probability",
            f"{probability * 100:.2f}%"
        )

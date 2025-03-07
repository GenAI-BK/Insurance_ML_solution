import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ✅ Load encoder, scaler, and trained model
with open("onehot_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load the dictionary containing multiple models
with open("trained_model.pkl", "rb") as f:
    models = pickle.load(f)

# Extract the "Logistic Regression" model
model = models["Logistic Regression"]

# ✅ Load dataset (for unique values)
df = pd.read_csv("Crosswalk2015.csv")

# ✅ Extract unique values for dropdowns
unique_values = {
    "ChildAdultOnly_2014": df["ChildAdultOnly_2014"].unique(),
    "MultistatePlan_2014": df["MultistatePlan_2014"].unique(),
    "MetalLevel_2014": df["MetalLevel_2014"].unique(),
    "ReasonForCrosswalk": df["ReasonForCrosswalk"].unique(),
    "CrosswalkLevel": df["CrosswalkLevel"].unique(),
    "IssuerID_2014": df["IssuerID_2014"].unique(),
    "PlanID_2014": df["PlanID_2014"].unique()
}

# ✅ Define mapping for dropdowns
child_adult_mapping = {
    0: "single user",
    1: "spouse and child",
    2: "Dependent parents"
}

reason_for_crosswalk_mapping = {
    0: "No crosswalk necessary",
    1: "Merger or acquisition",
    2: "Plan modification",
    3: "Regulatory requirement",
    4: "Product discontinuation",
    5: "Network change",
    6: "Other reasons"
}

crosswalk_level_mapping = {
    0: "Issuer Level",
    1: "Plan Level",
    2: "Benefit Level",
    3: "Network Level",
    4: "Regulatory Level"
}

issuer_id_to_company_name = {
    14002: "UnitedHealthcare",
    16842: "Blue Cross Blue Shield",
    17575: "Aetna",
    18558: "Cigna",
    33602: "Kaiser Permanente",
    35700: "Humana",
    36096: "Anthem",
    49046: "Molina Healthcare",
    71268: "WellCare",
    88380: "Centene Corporation"
}
st.set_page_config(page_title="Insurance Sell Prediction", page_icon="📊", layout="wide")

st.markdown(
    '''<style>
        body {
            background-color: #87CEEB;
            color: #333333;
        }
        h1, h2, h3, label {
            font-weight: bold;
            color: #333333;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
        }
.st-emotion-cache-18netey.egexzqm2 h1 {
    font-size: 1.75rem;
    padding-bottom: 30px;
    position: relative;
    padding-top:0;
}
.st-emotion-cache-89jlt8.egexzqm0 p {
    font-weight: 600;
}

.st-emotion-cache-18netey.egexzqm2 h1:after {
    content: "";
    position: absolute;
    background: #16c60c;
    width: 50px;
    height: 3px;
    left: 43px;
    top: 40px;
    
}
.st-emotion-cache-1104ytp h3 {
    font-size: 1.5rem;
    font-weight:800;
    
}
.stElementContainer.element-container.st-emotion-cache-1w8hq0f.e6rk8up1 h3{
    font-size: 1.75rem;
    padding-bottom: 30px;
    position: relative;
    padding-top:30px;
}
.stElementContainer.element-container.st-emotion-cache-1w8hq0f.e6rk8up1 h3:after{
    content: "";
    position: absolute;
    background: #16c60c;
    width: 50px;
    height: 3px;
    left: 43px;
    top: 70px;
}
.stButton button:hover, .stButton button:focus {
    background: #2e9332;
    color: #fff !important;
    border-color: transparent !important;
}
.stMainBlockContainer {
    background: #f3f3f3;
}
.stColumn.st-emotion-cache-wt9exi.e6rk8up2 {
    background: #fff;
    padding: 20px;
    box-shadow: 0 0 15px #ddd;
    border-radius: 10px;
}

.stHorizontalBlock.st-emotion-cache-ocqkz7:nth-child(1) .stColumn.st-emotion-cache-wt9exi.e6rk8up2 {
    background: none;
    box-shadow: none;
    border-radius: 0;
    padding: 0;
}
    </style>''',

    unsafe_allow_html=True

)
st.logo("bk-logo.png", size="medium", link=None, icon_image=None)
st.title("📊 Insurance Sell Prediction")

col_left, col_right = st.columns([1,1])

with col_left:
    # ✅ Collect user input with mapped labels
    col1, col2 = st.columns(2)
    with col1:
        plan_id = st.selectbox("USER ID", unique_values["PlanID_2014"])
        benefit_plan = st.selectbox(
            "Benefit Plan",
            options=list(child_adult_mapping.keys()),
            format_func=lambda x: child_adult_mapping[x]
        )
        crosswalk_level = st.selectbox(
        "Crosswalk Level",
        options=list(crosswalk_level_mapping.keys()),
        format_func=lambda x: crosswalk_level_mapping[x]
    )
        issuer_id = st.selectbox(
        "Issuer ID (Insurance Company)",
        options=list(issuer_id_to_company_name.keys()),
        format_func=lambda x: issuer_id_to_company_name[x]
    )
 
    with col2:
        multistate_plan = st.selectbox("Multistate Plan", unique_values["MultistatePlan_2014"])
        membership_tier = st.selectbox("Membership Tier", unique_values["MetalLevel_2014"])
        reason_for_crosswalk = st.selectbox(
        "Reason For Crosswalk",
        options=list(reason_for_crosswalk_mapping.keys()),
        format_func=lambda x: reason_for_crosswalk_mapping[x]
    )
 

   

    

    

    user_input = pd.DataFrame({
        "PlanID_2014": [plan_id],
        "ChildAdultOnly_2014": [benefit_plan],
        "MultistatePlan_2014": [multistate_plan],
        "MetalLevel_2014": [membership_tier],
        "ReasonForCrosswalk": [reason_for_crosswalk],
        "CrosswalkLevel": [crosswalk_level],
        "IssuerID_2014": [issuer_id]
    })



with col_right:
    user_input_cat = encoder.transform(user_input[["ChildAdultOnly_2014", "MultistatePlan_2014", "MetalLevel_2014"]])
    user_input_encoded = pd.DataFrame(user_input_cat, columns=encoder.get_feature_names_out())

    user_input_num = pd.DataFrame(
        scaler.transform(user_input[["ReasonForCrosswalk", "CrosswalkLevel", "IssuerID_2014"]]), 
        columns=["ReasonForCrosswalk", "CrosswalkLevel", "IssuerID_2014"]
    )

    user_input_final = pd.concat([user_input_encoded, user_input_num], axis=1)

    prediction_proba = model.predict_proba(user_input_final)[0]
    st.subheader("Prediction Probabilities")
    labels = ["Discontinue", "Continued"]
    fig, ax = plt.subplots()
    sns.barplot(x=labels, y=prediction_proba * 100, palette="pastel")
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    for i, v in enumerate(prediction_proba * 100):
        ax.text(i, v + 2, f"{v:.2f}%", ha="center", fontsize=12)
    st.pyplot(fig)

# Visualization function with enhanced styling
def visualize_data(df):
    st.subheader("📈 Dashboard Visualizations")

    # Create a grid layout for better presentation
    col1, col2 = st.columns([1, 1])

    with col1:
        plt.figure(figsize=(6, 4))
        df['ChildAdultOnly_2014'] = df['ChildAdultOnly_2014'].map(child_adult_mapping)
        sns.countplot(x=df['ChildAdultOnly_2014'], palette="Purples")
        plt.title("Benefit Plan", fontsize=12)
        plt.xlabel("Status", fontsize=10)
        plt.ylabel("Count", fontsize=10)
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())

    with col2:
        plt.figure(figsize=(4, 3))
        top_issuers = df["IssuerID_2014"].value_counts().nlargest(10)
        issuer_names = [issuer_id_to_company_name.get(i, str(i)) for i in top_issuers.index]
        sns.barplot(x=issuer_names, y=top_issuers.values, palette="viridis")
        plt.title("Top 10 Issuers(names)", fontsize=12)
        plt.xlabel("Issuer Name", fontsize=8)
        plt.ylabel("Plans", fontsize=10)
        plt.xticks(rotation=45,fontsize=6)
        st.pyplot(plt.gcf())

   

    
    col3, col4 = st.columns([1, 1])

    if "State" in df.columns:
        with col3:
            plt.figure(figsize=(6, 4))
            state_counts = df["State"].value_counts().nlargest(15)
            sns.barplot(x=state_counts.index, y=state_counts.values, palette="Blues_r")
            plt.title("Top 15 States by Plans")
            plt.xlabel("State")
            plt.ylabel("Plans")
            plt.xticks(rotation=45)
            st.pyplot(plt.gcf())

    with col4:
        plt.figure(figsize=(2,2))
        df['MetalLevel_2014'].value_counts().plot.pie(
            autopct='%1.1f%%', cmap="Pastel1", startangle=90, 
            textprops={'fontsize': 3},
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
        plt.title("Membership Tier", fontsize=4)
        plt.ylabel('')
        
        st.pyplot(plt.gcf())


if st.button("📊 Show Dashboard"):
    visualize_data(df)    

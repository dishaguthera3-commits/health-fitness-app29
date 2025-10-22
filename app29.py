import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# ----------------- Page config -----------------
st.set_page_config(
    page_title="💚 SDG3 Health & Diet Recommender",
    layout="wide",
    page_icon="💪"
)

st.title("💚 AI SDG3 Health & Wellbeing Recommender")
st.markdown("Personalized diet, exercise, and lifestyle guidance powered by AI, aligned with **Sustainable Development Goal 3**.")

# ----------------- Load training CSV -----------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai_health_training_data_250.csv")
    le_gender = LabelEncoder()
    le_exercise = LabelEncoder()
    df['gender_enc'] = le_gender.fit_transform(df['gender'])
    df['exercise_enc'] = le_exercise.fit_transform(df['exercise_level'])
    return df, le_gender, le_exercise

try:
    data, le_gender, le_exercise = load_data()
except FileNotFoundError:
    st.error("CSV file 'ai_health_training_data_250.csv' not found! Make sure it is in the same folder as app.py.")
    st.stop()

# ----------------- Train Models -----------------
@st.cache_resource
def train_models(df):
    X = df[['weight','height','age','water_cups','gender_enc','exercise_enc']]
    
    reg_calories = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, df['calories'])
    reg_protein = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, df['protein_g'])
    reg_fat = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, df['fat_g'])
    reg_carb = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, df['carb_g'])
    
    clf_cardio = DecisionTreeClassifier(random_state=42).fit(X, df['cardio_plan'])
    clf_strength = DecisionTreeClassifier(random_state=42).fit(X, df['strength_plan'])
    clf_mobility = DecisionTreeClassifier(random_state=42).fit(X, df['mobility_plan'])
    
    return reg_calories, reg_protein, reg_fat, reg_carb, clf_cardio, clf_strength, clf_mobility

reg_calories, reg_protein, reg_fat, reg_carb, clf_cardio, clf_strength, clf_mobility = train_models(data)

# ----------------- Sidebar Inputs -----------------
st.sidebar.header("Enter Your Details")
weight = st.sidebar.number_input("Weight (kg)", 20.0, 200.0, 70.0)
height = st.sidebar.number_input("Height (cm)", 100.0, 220.0, 170.0)
age = st.sidebar.number_input("Age", 18, 80, 25)
gender = st.sidebar.selectbox("Gender", ["male","female"])
water_cups = st.sidebar.number_input("Water intake (cups/day)", 0, 20, 6)
exercise_level = st.sidebar.selectbox("Exercise level", ["sedentary","light","moderate","active","very active"])
goal = st.sidebar.radio("Goal", ["maintain", "lose", "gain"], index=0)

# ----------------- Prepare input for prediction -----------------
gender_enc = le_gender.transform([gender])[0]
exercise_enc = le_exercise.transform([exercise_level])[0]
input_df = pd.DataFrame([[weight,height,age,water_cups,gender_enc,exercise_enc]],
                        columns=['weight','height','age','water_cups','gender_enc','exercise_enc'])

# ----------------- BMI -----------------
bmi = weight / ((height/100)**2)
if bmi < 18.5:
    bmi_cat = "Underweight"
elif bmi < 25:
    bmi_cat = "Normal"
elif bmi < 30:
    bmi_cat = "Overweight"
else:
    bmi_cat = "Obese"

# ----------------- Predict Diet -----------------
pred_calories = reg_calories.predict(input_df)[0]
pred_protein = reg_protein.predict(input_df)[0]
pred_fat = reg_fat.predict(input_df)[0]
pred_carb = reg_carb.predict(input_df)[0]

# ----------------- Predict Exercise -----------------
pred_cardio = clf_cardio.predict(input_df)[0]
pred_strength = clf_strength.predict(input_df)[0]
pred_mobility = clf_mobility.predict(input_df)[0]

# ----------------- Tabs -----------------
tabs = st.tabs(["Body Metrics","Diet","Exercise","Hydration & Lifestyle"])

with tabs[0]:
    st.subheader("📊 Your Body Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Weight (kg)", weight)
    col2.metric("Height (cm)", height)
    col3.metric("BMI", f"{bmi:.1f} ({bmi_cat})")
    
    # BMI bar chart
    fig, ax = plt.subplots(figsize=(8,1))
    ranges = [(10,18.5,'Under'), (18.5,25,'Normal'), (25,30,'Over'), (30,40,'Obese')]
    colors = ['#ffd1dc','#c8f7c5','#fff2b2','#ffb3b3']
    for (start,end,_),c in zip(ranges,colors):
        ax.fill_between([start,end],[0,0],[1,1], color=c)
    ax.plot([bmi,bmi],[0,1], color='black', linewidth=3)
    ax.set_yticks([])
    ax.set_xlabel("BMI Scale")
    st.pyplot(fig)

with tabs[1]:
    st.subheader("🥗 Diet Recommendation")
    st.metric("Calories/day", f"{round(pred_calories)} kcal")
    st.write(f"Protein: {round(pred_protein)} g | Fat: {round(pred_fat)} g | Carbs: {round(pred_carb)} g")
    
    # Macronutrient pie chart
    fig2, ax2 = plt.subplots()
    ax2.pie([pred_protein*4, pred_fat*9, pred_carb*4], labels=['Protein','Fat','Carbs'], autopct='%1.1f%%', colors=['#4daf4a','#e41a1c','#377eb8'])
    ax2.set_title("Macronutrient Distribution")
    st.pyplot(fig2)
    
    with st.expander("Sample Meals 🍳"):
        st.write("**Breakfast:** Oats, eggs, fruit, nuts")
        st.write("**Lunch:** Brown rice/quinoa, lean protein, vegetables")
        st.write("**Snack:** Yogurt or fruit with nuts")
        st.write("**Dinner:** Salad with protein, small portion of carbs")

with tabs[2]:
    st.subheader("🏋️ Exercise Recommendation")
    col1, col2, col3 = st.columns(3)
    col1.info(f"Cardio:\n{pred_cardio}")
    col2.success(f"Strength:\n{pred_strength}")
    col3.warning(f"Mobility:\n{pred_mobility}")
    
    with st.expander("Additional Exercise Tips 🏃"):
        st.write("- Mix cardio and strength workouts.")
        st.write("- Stretch or do yoga post-workout.")
        st.write("- Start slow if new to training.")

with tabs[3]:
    st.subheader("💧 Hydration & Lifestyle")
    recommended_ml = 30*weight
    recommended_cups = round(recommended_ml/250)
    st.write(f"You drink: {water_cups} cups ({water_cups*250} ml)")
    st.write(f"Recommended: ~{recommended_cups} cups ({recommended_ml} ml)")
    if water_cups < recommended_cups:
        st.warning("Increase your water intake to stay hydrated!")
    else:
        st.success("Hydration is sufficient 👍")
    
    st.markdown("### 🌱 Lifestyle Tips")
    st.markdown("""
    - Sleep 7-9 hours per night  
    - Reduce processed foods and sugar  
    - Manage stress via meditation or walks  
    - Track progress but focus on healthy habits
    """)

st.markdown("---")
st.caption("This app is aligned with SDG 3 — Good Health & Wellbeing. AI-based recommendations are general guidance, not medical advice.")
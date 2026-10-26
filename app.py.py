import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

st.set_page_config(
    page_title="AI Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI Student Academic Performance & Placement Predictor")
st.markdown("Predict placement outcomes based on academic metrics, coding achievements, and soft skills using Machine Learning.")

# Load Model & Scaler
@st.cache_resource
def load_artifacts():
    if os.path.exists('models/placement_model.pkl') and os.path.exists('models/scaler.pkl'):
        model = joblib.load('models/placement_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    return None, None

model, scaler = load_artifacts()

if model is None or scaler is None:
    st.error("⚠️ Trained model not found! Please run `python src/generate_data.py` and `python src/train_model.py` first.")
    st.stop()

# Sidebar Input Options
st.sidebar.header("📋 Enter Student Profile")

cgpa = st.sidebar.slider("CGPA (out of 10.0)", 5.0, 10.0, 7.5, 0.1)
dsa_problems = st.sidebar.number_input("DSA Problems Solved (LeetCode/Codewars/HackerRank)", 0, 600, 180, 10)
projects_count = st.sidebar.slider("Completed Projects Count", 0, 10, 3)
internships = st.sidebar.selectbox("Internship Experience", [0, 1, 2, 3])
soft_skills_score = st.sidebar.slider("Soft Skills & Interview Rating (%)", 40, 100, 75, 1)

input_data = pd.DataFrame({
    'cgpa': [cgpa],
    'dsa_problems_solved': [dsa_problems],
    'projects_count': [projects_count],
    'internships': [internships],
    'soft_skills_score': [soft_skills_score]
})

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 Prediction Results")
    if st.button("Predict Placement Status", type="primary"):
        scaled_features = scaler.transform(input_data)
        prediction = model.predict(scaled_features)[0]
        prob = model.predict_proba(scaled_features)[0][1] * 100
        
        if prediction == 1:
            st.success(f"🎉 **High Probability of Placement!** ({prob:.1f}%)")
            st.balloons()
        else:
            st.warning(f"⚠️ **Placement Risk Detected.** Estimated Chance: {prob:.1f}%")
            
        st.write("---")
        st.markdown("### 💡 Recommended Improvement Plan")
        if cgpa < 7.5:
            st.write("• Focus on improving overall academic scores above **7.5 CGPA**.")
        if dsa_problems < 250:
            st.write("• Target solving **250+ DSA problems** across platforms like HackerRank or LeetCode.")
        if projects_count < 3:
            st.write("• Build at least **3 full-stack/AI projects** for your GitHub portfolio.")

with col2:
    st.subheader("📊 Skill Matrix Profile")
    metrics_df = pd.DataFrame({
        'Metric': ['CGPA (Scaled)', 'DSA (Scaled)', 'Projects (Scaled)', 'Internships (Scaled)', 'Soft Skills'],
        'Score': [
            (cgpa / 10.0) * 100,
            min((dsa_problems / 450.0) * 100, 100),
            min((projects_count / 6.0) * 100, 100),
            min((internships / 2.0) * 100, 100),
            soft_skills_score
        ]
    })
    
    fig = px.radial_plot(
        metrics_df, 
        r='Score', 
        theta='Metric', 
        line_close=True,
        title="Student Readiness Scorecard"
    )
    st.plotly_chart(fig, use_container_width=True)

st.write("---")
st.caption("Machine Learning Engine built with Scikit-Learn Random Forest Classifier.")
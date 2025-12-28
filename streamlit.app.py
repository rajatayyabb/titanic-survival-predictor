import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# Debug: Show current directory and files
def check_files():
    current_dir = os.getcwd()
    files = os.listdir(current_dir)
    return current_dir, files

# Load the model
@st.cache_resource
def load_model():
    try:
        # Check if files exist
        if not os.path.exists('random_forest_model.pkl'):
            st.error(f"❌ File 'random_forest_model.pkl' not found!")
            st.info(f"Current directory: {os.getcwd()}")
            st.info(f"Files in directory: {os.listdir('.')}")
            return None, None
        
        if not os.path.exists('feature_names.pkl'):
            st.error(f"❌ File 'feature_names.pkl' not found!")
            return None, None
            
        with open('random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, features
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, feature_names = load_model()

# Title and Description
st.title("🚢 Titanic Survival Prediction")
st.markdown("### Predict survival on the Titanic using Random Forest Classifier")
st.markdown("---")

if model is None or feature_names is None:
    st.error("⚠️ Model files not found! Please ensure 'random_forest_model.pkl' and 'feature_names.pkl' are in the same directory.")
    
    # Show debug info
    with st.expander("🔍 Debug Information"):
        current_dir, files = check_files()
        st.write(f"**Current Directory:** `{current_dir}`")
        st.write(f"**Files Found:** {files}")
        st.write("**Expected Files:**")
        st.write("- random_forest_model.pkl")
        st.write("- feature_names.pkl")
    st.stop()

# Sidebar for input
st.sidebar.header("🔍 Enter Passenger Details")

# Input fields
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3], help="1 = First, 2 = Second, 3 = Third")
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
age = st.sidebar.slider("Age", 0, 100, 25)
sibsp = st.sidebar.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.sidebar.number_input("Fare", 0.0, 600.0, 50.0, step=0.5)
embarked = st.sidebar.selectbox("Port of Embarkation", ["C (Cherbourg)", "Q (Queenstown)", "S (Southampton)"])

# Encode inputs
sex_encoded = 1 if sex == "Male" else 0
embarked_dict = {"C (Cherbourg)": 0, "Q (Queenstown)": 1, "S (Southampton)": 2}
embarked_encoded = embarked_dict[embarked]

# Create input dataframe
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_encoded],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked_encoded]
})

# Display input data
st.subheader("📋 Passenger Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Passenger Class", pclass)
    st.metric("Sex", sex)
    st.metric("Age", age)

with col2:
    st.metric("Siblings/Spouses", sibsp)
    st.metric("Parents/Children", parch)

with col3:
    st.metric("Fare", f"${fare:.2f}")
    st.metric("Embarked", embarked.split()[0])

st.markdown("---")

# Prediction
if st.button("🔮 Predict Survival", type="primary", use_container_width=True):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        st.subheader("🎯 Prediction Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.success("### ✅ SURVIVED")
                st.balloons()
            else:
                st.error("### ❌ NOT SURVIVED")
        
        with col2:
            st.info(f"**Survival Probability:** {probability[1]*100:.2f}%")
            st.info(f"**Death Probability:** {probability[0]*100:.2f}%")
        
        # Progress bar for probability
        st.markdown("#### Confidence Level")
        st.progress(float(probability[1]))
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Additional Information
st.markdown("---")
st.subheader("ℹ️ About This App")

with st.expander("Model Information"):
    st.write("""
    - **Algorithm:** Random Forest Classifier
    - **Dataset:** Titanic Dataset from Kaggle
    - **Features Used:** Passenger Class, Sex, Age, Siblings/Spouses, Parents/Children, Fare, Embarked
    - **Model Performance:** High accuracy on test data
    """)

with st.expander("How to Use"):
    st.write("""
    1. Enter passenger details in the sidebar
    2. Click the "Predict Survival" button
    3. View the prediction result and probability
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | Random Forest Classifier</p>
    </div>
    """,
    unsafe_allow_html=True
)

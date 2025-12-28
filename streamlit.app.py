import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier

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

# Load the model - FIXED VERSION
@st.cache_resource
def load_model():
    try:
        # Check if files exist
        if not os.path.exists('random_forest_model.pkl'):
            st.error(f"❌ File 'random_forest_model.pkl' not found!")
            st.info(f"Current directory: {os.getcwd()}")
            st.info(f"Files in directory: {os.listdir('.')}")
            return None, None
        
        # First, let's see what's in the pickle file
        with open('random_forest_model.pkl', 'rb') as f:
            # Try to load and see the structure
            import pickle
            data = pickle.load(f)
            
        st.info(f"Loaded data type: {type(data)}")
        
        # If it's already a model, use it directly
        if isinstance(data, RandomForestClassifier):
            model = data
        else:
            # If it's a numpy array or other structure, try to reconstruct
            st.info(f"Data shape: {data.shape if hasattr(data, 'shape') else 'No shape'}")
            st.info(f"Data dtype: {data.dtype if hasattr(data, 'dtype') else 'No dtype'}")
            
            # Create a new Random Forest model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # If you have feature_names.pkl, load it
            if os.path.exists('feature_names.pkl'):
                with open('feature_names.pkl', 'rb') as f:
                    features = pickle.load(f)
                return model, features
            else:
                # Use default feature names
                features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
                return model, features
                
        # Load features if available
        if os.path.exists('feature_names.pkl'):
            with open('feature_names.pkl', 'rb') as f:
                features = pickle.load(f)
        else:
            features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
            
        return model, features
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        # Return a default model as fallback
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        return model, features

model, feature_names = load_model()

# Title and Description
st.title("🚢 Titanic Survival Prediction")
st.markdown("### Predict survival on the Titanic using Random Forest Classifier")
st.markdown("---")

# Check if we have a valid model
if model is None:
    st.error("⚠️ Failed to load model!")
    st.stop()

# Show model info
st.sidebar.info(f"Model: {type(model).__name__}")
st.sidebar.info(f"Features: {len(feature_names)}")

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

# Alternative: Create a simple model if pickle fails
@st.cache_resource
def create_fallback_model():
    """Create a simple fallback model for demonstration"""
    from sklearn.ensemble import RandomForestClassifier
    # Create a simple model with reasonable defaults
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Create synthetic training data for Titanic-like patterns
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data with Titanic-like patterns
    X_train = np.column_stack([
        np.random.choice([1, 2, 3], n_samples),  # Pclass
        np.random.choice([0, 1], n_samples),     # Sex (0=Female, 1=Male)
        np.random.randint(0, 80, n_samples),     # Age
        np.random.randint(0, 5, n_samples),      # SibSp
        np.random.randint(0, 5, n_samples),      # Parch
        np.random.uniform(0, 600, n_samples),    # Fare
        np.random.choice([0, 1, 2], n_samples)   # Embarked
    ])
    
    # Survival probabilities based on Titanic patterns
    # Women, children, and first class more likely to survive
    survival_probs = (
        (X_train[:, 0] == 1) * 0.6 +  # First class
        (X_train[:, 0] == 2) * 0.4 +  # Second class
        (X_train[:, 0] == 3) * 0.2 +  # Third class
        (X_train[:, 1] == 0) * 0.7 +  # Female
        (X_train[:, 2] < 18) * 0.5 +  # Children
        (X_train[:, 5] > 100) * 0.3   # Higher fare
    ) / 3.0
    
    y_train = (survival_probs > 0.5).astype(int)
    
    model.fit(X_train, y_train)
    return model

# Check if we should use fallback
use_fallback = st.sidebar.checkbox("Use fallback model (if pickle fails)", value=False)

# Prediction
if st.button("🔮 Predict Survival", type="primary", use_container_width=True):
    try:
        if use_fallback:
            fallback_model = create_fallback_model()
            prediction = fallback_model.predict(input_data)[0]
            probability = fallback_model.predict_proba(input_data)[0]
            st.sidebar.warning("Using fallback model for prediction")
        else:
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
        st.info("Try enabling the fallback model option in the sidebar")

# Additional Information
st.markdown("---")
st.subheader("ℹ️ About This App")

with st.expander("Debug Information"):
    current_dir, files = check_files()
    st.write(f"**Current Directory:** `{current_dir}`")
    st.write(f"**Files Found:** {files}")
    st.write("**Expected Files:**")
    st.write("- random_forest_model.pkl")
    st.write("- feature_names.pkl")

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
    2. If pickle file fails, enable "Use fallback model" option
    3. Click the "Predict Survival" button
    4. View the prediction result and probability
    """)

with st.expander("Troubleshooting"):
    st.write("""
    **Common Issues:**
    1. **Pickle version mismatch:** The model was saved with a different scikit-learn version
    2. **File not found:** Ensure both .pkl files are in the same directory
    3. **Corrupted pickle file:** The pickle file might be corrupted
    
    **Solutions:**
    1. Use the fallback model option
    2. Re-train and save the model with current scikit-learn version
    3. Check file permissions and paths
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | Random Forest Classifier</p>
        <p><small>Note: If pickle file fails, a fallback model will be used</small></p>
    </div>
    """,
    unsafe_allow_html=True
)

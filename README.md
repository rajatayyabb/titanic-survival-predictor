# 🚢 Titanic Survival Predictor

A machine learning web application that predicts passenger survival on the Titanic using Random Forest Classifier.

## 🌟 Features

- **Interactive UI**: Easy-to-use interface built with Streamlit
- **Real-time Predictions**: Get instant survival predictions
- **Probability Display**: See confidence levels for predictions
- **Visual Feedback**: Intuitive design with emojis and progress bars

## 🚀 Live Demo

[Try the app here](https://your-app-name.streamlit.app)

## 📊 Model Performance

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 82.12%
- **Precision**: 81.25%
- **Recall**: 80.54%
- **F1-Score**: 80.89%

## 🛠️ Technologies Used

- Python 3.11
- Streamlit
- Scikit-learn
- Pandas
- NumPy

## 📁 Project Structure

```
titanic-survival-predictor/
│
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── random_forest_model.pkl     # Trained model
├── feature_names.pkl           # Feature names
└── README.md                   # Project documentation
```

## 🏃‍♂️ Run Locally

1. Clone the repository:
```bash
git clone https://github.com/YOUR-USERNAME/titanic-survival-predictor.git
cd titanic-survival-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

4. Open your browser and go to `http://localhost:8501`

## 📝 How to Use

1. Enter passenger details in the sidebar:
   - Passenger Class (1st, 2nd, or 3rd)
   - Sex (Male or Female)
   - Age (0-100)
   - Number of Siblings/Spouses aboard
   - Number of Parents/Children aboard
   - Fare amount
   - Port of Embarkation

2. Click the "Predict Survival" button

3. View the prediction result and survival probability

## 📊 Dataset

The model is trained on the famous [Titanic Dataset from Kaggle](https://www.kaggle.com/c/titanic), which contains passenger information including:
- Demographics (age, sex)
- Ticket class
- Family relations
- Fare paid
- Port of embarkation

## 🧠 Model Details

- **Algorithm**: Random Forest Classifier
- **Number of Trees**: 100
- **Max Depth**: 10
- **Features Used**: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

## 👨‍💻 Author

Your Name - Lab 12 Machine Learning Project

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/c/titanic)
- Built with [Streamlit](https://streamlit.io/)
- Machine learning with [Scikit-learn](https://scikit-learn.org/)

---

**Built with ❤️ using Random Forest Classifier**

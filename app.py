import joblib
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered"
)

st.title("🚢 Titanic Passenger Survival Predictor")
st.write(
    "Enter passenger details below to predict their survival probability using our trained ML Pipeline."
)

# Load Trained Pipeline
@st.cache_resource
def load_model():
  return joblib.load("titanic_pipeline.joblib")


try:
  pipeline = load_model()
except Exception as e:
  st.error(f"Error loading model pipeline: {e}")

# Sidebar / Main Form Inputs
st.header("Passenger Attributes")

col1, col2 = st.columns(2)

with col1:
  pclass = st.selectbox(
      "Passenger Class (Pclass)",
      [1, 2, 3],
      index=2,
      help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class",
  )
  sex = st.selectbox("Gender", ["male", "female"])
  age = st.slider("Age", min_value=1, max_value=80, value=28)
  embarked = st.selectbox(
      "Port of Embarkation",
      ["S", "C", "Q"],
      format_func=lambda x: {
          "S": "Southampton",
          "C": "Cherbourg",
          "Q": "Queenstown",
      }[x],
  )

with col2:
  fare = st.number_input("Ticket Fare ($)", min_value=0.0, value=32.2, step=5.0)
  sibsp = st.number_input(
      "Siblings / Spouses Aboard (SibSp)", min_value=0, max_value=8, value=0
  )
  parch = st.number_input(
      "Parents / Children Aboard (Parch)", min_value=0, max_value=6, value=0
  )

# Calculate Engineered Features inside the app
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# Predict Button
if st.button("🔮 Predict Survival", type="primary"):
  # Create DataFrame with exact column names expected by the pipeline
  input_data = pd.DataFrame([{
      "Age": age,
      "Fare": fare,
      "FamilySize": family_size,
      "Sex": sex,
      "Embarked": embarked,
      "Pclass": pclass,
      "IsAlone": is_alone,
  }])

  # Make Prediction
  prediction = pipeline.predict(input_data)[0]
  proba = pipeline.predict_proba(input_data)[0]

  st.divider()
  if prediction == 1:
    st.success(
        f"🎉 **Predicted Outcome: SURVIVED**\n\nSurvival Confidence:"
        f" **{proba[1]*100:.1f}%**"
    )
  else:
    st.error(
        f"💀 **Predicted Outcome: DID NOT SURVIVE**\n\nSurvival Confidence:"
        f" **{proba[0]*100:.1f}%**"
    )
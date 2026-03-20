import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import shap

# Load model
model = joblib.load("gbr_model.pkl")
expected_features = model.feature_names_in_

st.title("Hospital Equipment Remaining Useful Life Prediction")

# Sidebar inputs
equipment_age = st.sidebar.number_input("Equipment Age (Years)", 0, 20, 5)
maintenance_count = st.sidebar.number_input("Maintenance Count", 0, 20, 10)
downtime_duration = st.sidebar.number_input("Downtime Duration (Hours)", 0, 120, 50)
error_count = st.sidebar.number_input("Error Count", 0, 50, 5)

equipment_type = st.sidebar.selectbox(
    "Equipment Type",
    [
        "Anesthesia_Machine",
        "MRI",
        "X-Ray",
        "Ultrasound",
        "Defibrillator",
        "ECG",
        "Ventilator"
    ]
)

# Base input
input_data = pd.DataFrame({
    "Equipment_Age_Years": [equipment_age],
    "Maintenance_Count": [maintenance_count],
    "Downtime_Duration": [downtime_duration],
    "Error_Count": [error_count],
})

# Add missing columns
for col in expected_features:
    if col not in input_data.columns:
        input_data[col] = 0

# Set equipment flag
equipment_col = f"Equipment_Type_{equipment_type}"
if equipment_col in input_data.columns:
    input_data[equipment_col] = 1
# Reorder
input_data = input_data[expected_features]

# Prediction
prediction = model.predict(input_data)

if st.button("Predict Remaining Useful Life"):
    st.success(f"Predicted Remaining Useful Life: **{prediction[0]:.2f} Years**")

st.subheader("Input Data")
st.write(input_data)

# Risk
st.subheader("Risk Assessment")
if prediction[0] < 2 or error_count > 40 or maintenance_count >10 :
    st.error("High Risk of Equipment Failure!")
elif prediction[0] < 5 or error_count > 20:
    st.warning("Moderate Risk of Equipment Failure!")
else:
    st.success("Low Risk of Equipment Failure!")

# Feature groups
non_equipment_features = [
    c for c in input_data.columns if not c.startswith("Equipment_Type_")
]

# Tabs
tab1, tab2 = st.tabs(["Graphical Overview", "SHAP Explainability"])

# TAB 1
with tab1:
    fig, ax = plt.subplots()
    ax.barh(non_equipment_features, input_data[non_equipment_features].iloc[0])
    ax.set_title("Equipment Condition Overview")
    st.pyplot(fig)

# TAB 2
with tab2:
    st.subheader("Why Did the Model Predict This?")

    st.write(
        """
        This chart explains **how operational and maintenance factors**
        influenced the Remaining Useful Life prediction.
        """
    )

    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_data)

    # Get base value safely
    base_value = explainer.expected_value
    if isinstance(base_value, (list, tuple)):
        base_value = base_value[0]

    # ❗ Remove equipment-type columns ONLY for display
    display_features = [
        col for col in input_data.columns
        if not col.startswith("Equipment_Type_")
    ]

    shap_display_values = shap_values.values[0][
        [input_data.columns.get_loc(c) for c in display_features]
    ]

    display_data = input_data[display_features].iloc[0]

    # Waterfall plot
    fig = plt.figure()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_display_values,
            base_values=base_value,
            feature_names=display_features,
            data=display_data.values
        ),
        show=False
    )

    st.pyplot(fig)
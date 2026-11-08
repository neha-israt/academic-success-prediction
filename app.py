import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Academic Success Predictor",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# LOAD DATA AND MODELS
# --------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv(
        BASE_DIR / "data.csv",
        sep=";"
    )

    data.columns = data.columns.str.strip()

    return data


@st.cache_resource
def load_models():
    preprocessor = joblib.load(
        BASE_DIR / "preprocessor.pkl"
    )

    logistic_model = joblib.load(
        BASE_DIR / "logistic_regression_model.pkl"
    )

    balanced_model = joblib.load(
        BASE_DIR / "balanced_logistic_regression_model.pkl"
    )

    return preprocessor, logistic_model, balanced_model


data = load_data()

preprocessor, logistic_model, balanced_model = load_models()


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def get_column_options(column_name):

    if column_name not in data.columns:
        st.error(f"Column not found in dataset: {column_name}")
        st.stop()

    return sorted(
        data[column_name].dropna().unique().tolist()
    )


def categorical_input(label, column_name):

    options = get_column_options(column_name)

    return st.selectbox(
        label,
        options,
        key=f"input_{column_name}"
    )


def numerical_input(label, column_name):

    if column_name not in data.columns:
        st.error(f"Column not found in dataset: {column_name}")
        st.stop()

    minimum = float(data[column_name].min())
    maximum = float(data[column_name].max())
    average = float(data[column_name].median())

    return st.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        value=average,
        step=1.0,
        key=f"input_{column_name}"
    )


# --------------------------------------------------
# APP TITLE AND INTRODUCTION
# --------------------------------------------------

st.title("🎓 Academic Success Predictor")

st.write(
    """
    This application uses machine learning to estimate a student's
    academic outcome based on personal, admission, financial, and
    academic information.

    The model predicts one of three possible outcomes:
    **Graduate, Dropout, or Enrolled.**
    """
)

st.info(
    """
    **Important:** This model was trained using data from a Portuguese
    higher-education institution. Its predictions may not generalize
    to other countries, universities, or education systems.

    It also uses first- and second-semester academic performance,
    so it is intended for students whose records for those semesters
    are already available.

    Predictions are estimates for educational demonstration purposes,
    not guarantees or recommendations for high-stakes decisions.
    """
)

st.divider()


# --------------------------------------------------
# STUDENT INPUT FORM
# --------------------------------------------------

st.subheader("Student Information")

st.write(
    "Enter the student's information in the sections below."
)

input_data = {}


# --------------------------------------------------
# 1. STUDENT INFORMATION
# --------------------------------------------------

with st.expander("1. Student Information", expanded=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        input_data["Marital status"] = categorical_input(
            "Marital Status",
            "Marital status"
        )

        input_data["Application mode"] = categorical_input(
            "Application Mode",
            "Application mode"
        )

        input_data["Application order"] = numerical_input(
            "Application Order",
            "Application order"
        )

        input_data["Course"] = categorical_input(
            "Course",
            "Course"
        )

    with col2:

        input_data["Daytime/evening attendance"] = categorical_input(
            "Attendance Type",
            "Daytime/evening attendance"
        )

        input_data["Previous qualification"] = categorical_input(
            "Previous Qualification",
            "Previous qualification"
        )

        input_data["Previous qualification (grade)"] = numerical_input(
            "Previous Qualification Grade",
            "Previous qualification (grade)"
        )

        input_data["Nacionality"] = st.selectbox(
            "Nationality",
            ["Portuguese", "Other"],
            key="input_nationality"
        )

    with col3:

        input_data["Mother's qualification"] = categorical_input(
            "Mother's Qualification",
            "Mother's qualification"
        )

        input_data["Father's qualification"] = categorical_input(
            "Father's Qualification",
            "Father's qualification"
        )

        input_data["Mother's occupation"] = categorical_input(
            "Mother's Occupation",
            "Mother's occupation"
        )

        input_data["Father's occupation"] = categorical_input(
            "Father's Occupation",
            "Father's occupation"
        )


# --------------------------------------------------
# 2. ADMISSION INFORMATION
# --------------------------------------------------

with st.expander("2. Admission Information", expanded=False):

    col1, col2, col3 = st.columns(3)

    with col1:

        input_data["Admission grade"] = numerical_input(
            "Admission Grade",
            "Admission grade"
        )

    with col2:

        input_data["Displaced"] = categorical_input(
            "Displaced",
            "Displaced"
        )

        input_data["Educational special needs"] = categorical_input(
            "Educational Special Needs",
            "Educational special needs"
        )

    with col3:

        input_data["Debtor"] = categorical_input(
            "Debtor",
            "Debtor"
        )

        input_data["Tuition fees up to date"] = categorical_input(
            "Tuition Fees Up to Date",
            "Tuition fees up to date"
        )

        input_data["Scholarship holder"] = categorical_input(
            "Scholarship Holder",
            "Scholarship holder"
        )


# --------------------------------------------------
# 3. FINANCIAL INFORMATION
# --------------------------------------------------

with st.expander("3. Financial Information", expanded=False):

    col1, col2, col3 = st.columns(3)

    with col1:

        input_data["Unemployment rate"] = numerical_input(
            "Unemployment Rate",
            "Unemployment rate"
        )

    with col2:

        input_data["Inflation rate"] = numerical_input(
            "Inflation Rate",
            "Inflation rate"
        )

    with col3:

        input_data["GDP"] = numerical_input(
            "GDP",
            "GDP"
        )


# --------------------------------------------------
# 4. FIRST SEMESTER ACADEMIC PERFORMANCE
# --------------------------------------------------

with st.expander(
    "4. First Semester Academic Performance",
    expanded=False
):

    col1, col2, col3 = st.columns(3)

    with col1:

        input_data["Curricular units 1st sem (credited)"] = numerical_input(
            "Credited Units",
            "Curricular units 1st sem (credited)"
        )

        input_data["Curricular units 1st sem (enrolled)"] = numerical_input(
            "Enrolled Units",
            "Curricular units 1st sem (enrolled)"
        )

    with col2:

        input_data["Curricular units 1st sem (evaluations)"] = numerical_input(
            "Evaluated Units",
            "Curricular units 1st sem (evaluations)"
        )

        input_data["Curricular units 1st sem (approved)"] = numerical_input(
            "Approved Units",
            "Curricular units 1st sem (approved)"
        )

    with col3:

        input_data["Curricular units 1st sem (grade)"] = numerical_input(
            "Average Grade",
            "Curricular units 1st sem (grade)"
        )

        input_data["Curricular units 1st sem (without evaluations)"] = numerical_input(
            "Units Without Evaluations",
            "Curricular units 1st sem (without evaluations)"
        )


# --------------------------------------------------
# 5. SECOND SEMESTER ACADEMIC PERFORMANCE
# --------------------------------------------------

with st.expander(
    "5. Second Semester Academic Performance",
    expanded=False
):

    col1, col2, col3 = st.columns(3)

    with col1:

        input_data["Curricular units 2nd sem (credited)"] = numerical_input(
            "Credited Units",
            "Curricular units 2nd sem (credited)"
        )

        input_data["Curricular units 2nd sem (enrolled)"] = numerical_input(
            "Enrolled Units",
            "Curricular units 2nd sem (enrolled)"
        )

    with col2:

        input_data["Curricular units 2nd sem (evaluations)"] = numerical_input(
            "Evaluated Units",
            "Curricular units 2nd sem (evaluations)"
        )

        input_data["Curricular units 2nd sem (approved)"] = numerical_input(
            "Approved Units",
            "Curricular units 2nd sem (approved)"
        )

    with col3:

        input_data["Curricular units 2nd sem (grade)"] = numerical_input(
            "Average Grade",
            "Curricular units 2nd sem (grade)"
        )

        input_data["Curricular units 2nd sem (without evaluations)"] = numerical_input(
            "Units Without Evaluations",
            "Curricular units 2nd sem (without evaluations)"
        )


# --------------------------------------------------
# 6. OTHER INFORMATION
# --------------------------------------------------

with st.expander(
    "6. Other Information",
    expanded=False
):

    col1, col2, col3 = st.columns(3)

    with col1:

        input_data["Age at enrollment"] = numerical_input(
            "Age at Enrollment",
            "Age at enrollment"
        )

    with col2:

        input_data["International"] = categorical_input(
            "International Student",
            "International"
        )

    with col3:

        input_data["Gender"] = categorical_input(
            "Gender",
            "Gender"
        )


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

st.divider()

st.subheader("Prediction Settings")

selected_model = st.selectbox(
    "Choose a prediction model",
    [
        "Logistic Regression",
        "Balanced Logistic Regression"
    ],
    key="model_selection"
)

if selected_model == "Logistic Regression":
    model = logistic_model

else:
    model = balanced_model


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.divider()

if st.button(
    "Predict Academic Outcome",
    type="primary",
    use_container_width=True
):

    try:

        # Create DataFrame from user input
        input_df = pd.DataFrame([input_data])

        # Remove accidental whitespace from input names
        input_df.columns = input_df.columns.str.strip()

        # Get original feature names from saved preprocessor
        expected_features = list(
            preprocessor.feature_names_in_
        )

        # Map cleaned names to exact saved feature names
        feature_mapping = {
            str(feature).strip(): feature
            for feature in expected_features
        }

        # Restore original names, including tabs
        input_df = input_df.rename(
            columns={
                column: feature_mapping[column]
                for column in input_df.columns
                if column in feature_mapping
            }
        )

        # Ensure correct feature order
        input_df = input_df[expected_features]

        # Preprocess user input
        processed_input = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(processed_input)[0]

        # Get prediction probabilities
        probabilities = model.predict_proba(processed_input)[0]

        # Get class names
        class_names = model.classes_

        # Create probability DataFrame
        probability_df = pd.DataFrame({
            "Academic Outcome": class_names,
            "Probability": probabilities
        })

        probability_df["Probability (%)"] = (
            probability_df["Probability"] * 100
        ).round(2)

        # Sort probabilities
        probability_df = probability_df.sort_values(
            by="Probability",
            ascending=False
        )

        # Get probability of predicted class
        predicted_probability = max(probabilities) * 100

        # --------------------------------------------------
        # DISPLAY PREDICTION
        # --------------------------------------------------

        st.success("Prediction completed successfully!")

        st.subheader("Predicted Academic Outcome")

        st.markdown(
            f"# 🎓 {prediction}"
        )

        st.write(
            f"The model favors **{prediction}**, "
            "based on the information provided."
        )

        # --------------------------------------------------
        # PREDICTION STRENGTH
        # --------------------------------------------------

        st.subheader("Prediction Strength")

        st.metric(
            label="Probability assigned to predicted outcome",
            value=f"{predicted_probability:.2f}%"
        )

        if predicted_probability >= 85:

            st.success(
                "The model strongly favors this outcome."
            )

        elif predicted_probability >= 60:

            st.info(
                "The model moderately favors this outcome."
            )

        else:

            st.warning(
                "The model is less certain about this outcome."
            )

        st.caption(
            """
            These labels are based on probability thresholds and
            should not be interpreted as calibrated confidence.
            """
        )

        # --------------------------------------------------
        # HORIZONTAL PROBABILITY CHART
        # --------------------------------------------------

        st.divider()

        st.subheader("Prediction Probabilities")

        chart_data = probability_df.set_index(
            "Academic Outcome"
        )[["Probability"]]

        st.bar_chart(
            chart_data,
            horizontal=True
        )

        # --------------------------------------------------
        # OUTCOME BREAKDOWN
        # --------------------------------------------------

        st.subheader("Outcome Breakdown")

        display_df = probability_df[
            ["Academic Outcome", "Probability (%)"]
        ].copy()

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # EXPLANATION
        # --------------------------------------------------

        st.caption(
            """
            The probabilities represent the model's estimated support
            for each possible outcome based on the information entered.
            They are not guaranteed real-world chances of graduating,
            dropping out, or remaining enrolled.
            """
        )

    except Exception as e:

        st.error(
            "Something went wrong while generating the prediction."
        )

        st.exception(e)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Academic Success Prediction | Machine Learning Project"
)
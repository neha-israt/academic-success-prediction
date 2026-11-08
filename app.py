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
# CATEGORY LABELS
# These change only what users see.
# Original values are passed to the model.
# --------------------------------------------------

CATEGORY_LABELS = {

    "Marital status": {
        1: "Single",
        2: "Married",
        3: "Widower",
        4: "Divorced",
        5: "Facto union",
        6: "Legally separated"
    },

    "Application mode": {
        1: "1st phase - General contingent",
        2: "Ordinance No. 612/93",
        5: "1st phase - Special contingent (Azores)",
        7: "Holders of other higher courses",
        10: "Ordinance No. 854-B/99",
        15: "International student (Bachelor)",
        16: "1st phase - Special contingent (Madeira)",
        17: "2nd phase - General contingent",
        18: "3rd phase - General contingent",
        26: "Ordinance No. 533-A/99 (Different Plan)",
        27: "Ordinance No. 533-A/99 (Other Institution)",
        39: "Over 23 years old",
        42: "Transfer",
        43: "Change of course",
        44: "Technological specialization diploma holders",
        51: "Change of institution/course",
        53: "Short cycle diploma holders",
        57: "Change of institution/course (International)"
    },

    "Course": {
        33: "Biofuel Production Technologies",
        171: "Animation and Multimedia Design",
        8014: "Social Service (Evening)",
        9003: "Agronomy",
        9070: "Communication Design",
        9085: "Veterinary Nursing",
        9119: "Informatics Engineering",
        9130: "Equinculture",
        9147: "Management",
        9238: "Social Service",
        9254: "Tourism",
        9500: "Nursing",
        9556: "Oral Hygiene",
        9670: "Advertising and Marketing Management",
        9773: "Journalism and Communication",
        9853: "Basic Education",
        9991: "Management (Evening)"
    },

    "Daytime/evening attendance": {
        0: "Evening",
        1: "Daytime"
    },

    "Previous qualification": {
        1: "Secondary education",
        2: "Higher education - Bachelor's degree",
        3: "Higher education - Degree",
        4: "Higher education - Master's",
        5: "Higher education - Doctorate",
        6: "Frequency of higher education",
        9: "12th year of schooling - Not completed",
        10: "11th year of schooling - Not completed",
        12: "Other - 11th year of schooling",
        14: "10th year of schooling",
        15: "10th year of schooling - Not completed",
        19: "Basic education 3rd cycle",
        38: "Basic education 2nd cycle",
        39: "Technological specialization course",
        40: "Higher education - Degree (1st cycle)",
        42: "Professional higher technical course",
        43: "Higher education - Master's (2nd cycle)"
    },

    "Mother's qualification": {
        1: "Secondary education - 12th year",
        2: "Higher education - Bachelor's degree",
        3: "Higher education - Degree",
        4: "Higher education - Master's",
        5: "Higher education - Doctorate",
        6: "Frequency of higher education",
        9: "12th year - Not completed",
        10: "11th year - Not completed",
        11: "7th year (Old)",
        12: "Other - 11th year",
        14: "10th year",
        18: "General commerce course",
        19: "Basic education 3rd cycle",
        22: "Technical-professional course",
        26: "7th year of schooling",
        27: "2nd cycle of general high school",
        29: "9th year - Not completed",
        30: "8th year of schooling",
        34: "Unknown",
        35: "Cannot read or write",
        36: "Can read without 4th year",
        37: "Basic education 1st cycle",
        38: "Basic education 2nd cycle",
        39: "Technological specialization course",
        40: "Higher education - Degree",
        41: "Specialized higher studies course",
        42: "Professional higher technical course",
        43: "Higher education - Master's",
        44: "Higher education - Doctorate"
    },

    "Father's qualification": {
        1: "Secondary education - 12th year",
        2: "Higher education - Bachelor's degree",
        3: "Higher education - Degree",
        4: "Higher education - Master's",
        5: "Higher education - Doctorate",
        6: "Frequency of higher education",
        9: "12th year - Not completed",
        10: "11th year - Not completed",
        11: "7th year (Old)",
        12: "Other - 11th year",
        13: "2nd year complementary high school",
        14: "10th year",
        18: "General commerce course",
        19: "Basic education 3rd cycle",
        20: "Complementary high school course",
        22: "Technical-professional course",
        25: "Complementary high school - Not completed",
        26: "7th year of schooling",
        27: "2nd cycle of general high school",
        29: "9th year - Not completed",
        30: "8th year of schooling",
        31: "General administration and commerce",
        33: "Supplementary accounting and administration",
        34: "Unknown",
        35: "Cannot read or write",
        36: "Can read without 4th year",
        37: "Basic education 1st cycle",
        38: "Basic education 2nd cycle",
        39: "Technological specialization course",
        40: "Higher education - Degree",
        41: "Specialized higher studies course",
        42: "Professional higher technical course",
        43: "Higher education - Master's",
        44: "Higher education - Doctorate"
    }
}


# --------------------------------------------------
# OCCUPATION LABELS
# --------------------------------------------------

GENERAL_OCCUPATIONS = {
    0: "Student",
    1: "Legislative and executive directors",
    2: "Intellectual and scientific professionals",
    3: "Intermediate-level technicians",
    4: "Administrative staff",
    5: "Personal services and sales workers",
    6: "Farmers and skilled agricultural workers",
    7: "Industry and construction workers",
    8: "Machine operators and assemblers",
    9: "Unskilled workers",
    10: "Armed forces professions",
    90: "Other situation",
    99: "Not specified"
}

MOTHER_OCCUPATIONS = {
    **GENERAL_OCCUPATIONS,
    122: "Health professionals",
    123: "Teachers",
    125: "ICT specialists",
    131: "Science and engineering technicians",
    132: "Health associate professionals",
    134: "Legal, social and cultural technicians",
    141: "Office workers and secretaries",
    143: "Accounting and financial clerks",
    144: "Administrative support staff",
    151: "Personal service workers",
    152: "Sellers",
    153: "Personal care workers",
    171: "Skilled construction workers",
    173: "Printing and precision instrument workers",
    175: "Food processing and craft workers",
    191: "Cleaning workers",
    192: "Unskilled agricultural workers",
    193: "Unskilled industrial and construction workers",
    194: "Meal preparation assistants"
}

FATHER_OCCUPATIONS = {
    **GENERAL_OCCUPATIONS,
    101: "Armed forces officers",
    102: "Armed forces sergeants",
    103: "Other armed forces personnel",
    112: "Administrative and commercial directors",
    114: "Hotel and catering directors",
    121: "Physical science and engineering specialists",
    122: "Health professionals",
    123: "Teachers",
    124: "Finance and administration specialists",
    131: "Science and engineering technicians",
    132: "Health associate professionals",
    134: "Legal, social and cultural technicians",
    135: "ICT technicians",
    141: "Office workers and secretaries",
    143: "Accounting and financial clerks",
    144: "Administrative support staff",
    151: "Personal service workers",
    152: "Sellers",
    153: "Personal care workers",
    154: "Security service personnel",
    161: "Market-oriented farmers",
    163: "Subsistence farmers and fishermen",
    171: "Skilled construction workers",
    172: "Metal workers",
    174: "Electrical and electronics workers",
    175: "Food processing and craft workers",
    181: "Fixed plant and machine operators",
    182: "Assembly workers",
    183: "Vehicle drivers",
    192: "Unskilled agricultural workers",
    193: "Unskilled industrial and construction workers",
    194: "Meal preparation assistants",
    195: "Street vendors and service providers"
}

CATEGORY_LABELS["Mother's occupation"] = MOTHER_OCCUPATIONS
CATEGORY_LABELS["Father's occupation"] = FATHER_OCCUPATIONS


# --------------------------------------------------
# YES / NO AND OTHER BINARY LABELS
# --------------------------------------------------

YES_NO_COLUMNS = [
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
    "International"
]

for column in YES_NO_COLUMNS:
    CATEGORY_LABELS[column] = {
        0: "No",
        1: "Yes"
    }

CATEGORY_LABELS["Gender"] = {
    0: "Female",
    1: "Male"
}


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


def format_category(value, mapping):

    try:
        numeric_value = int(value)

        if numeric_value in mapping:
            return mapping[numeric_value]

    except (ValueError, TypeError):
        pass

    return f"Other category (code: {value})"


def categorical_input(label, column_name):

    options = get_column_options(column_name)

    mapping = CATEGORY_LABELS.get(column_name, {})

    return st.selectbox(
        label,
        options,
        format_func=lambda value: format_category(
            value,
            mapping
        ),
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

        # Clean input column names
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
            st.success("The model strongly favors this outcome.")

        elif predicted_probability >= 60:
            st.info("The model moderately favors this outcome.")

        else:
            st.warning("The model is less certain about this outcome.")

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
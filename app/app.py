
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="EstateIQ | Investment Advisor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================
# CUSTOM DESIGN
# ==================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

.stApp {
    background-color: #F5F4F0;
    color: #17263D;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

h1, h2, h3 {
    color: #17263D !important;
}

.hero {
    background: linear-gradient(120deg, #14243B, #243B57);
    padding: 36px 40px;
    border-radius: 18px;
    color: white;
    margin-bottom: 28px;
}

.hero h1 {
    color: #F1D49A !important;
    font-family: 'Playfair Display', serif;
    font-size: 38px;
    margin-bottom: 8px;
}

.hero p {
    color: #E1E7EF;
    font-size: 15px;
}

.section-heading {
    font-family: 'Playfair Display', serif;
    color: #17263D;
    font-size: 25px;
    font-weight: 700;
    margin-top: 12px;
}

.metric-card {
    background: white;
    border: 1px solid #E7E3D9;
    border-radius: 14px;
    padding: 20px;
    min-height: 115px;
    box-shadow: 0 3px 12px rgba(23,38,61,0.04);
}

.metric-label {
    color: #778397;
    font-size: 13px;
    margin-bottom: 10px;
}

.metric-value {
    color: #17263D;
    font-size: 25px;
    font-weight: 700;
}

.result-card {
    background: white;
    border: 1px solid #E7E3D9;
    border-radius: 16px;
    padding: 25px;
    min-height: 145px;
}

.result-label {
    color: #778397;
    font-size: 13px;
}

.result-value {
    color: #17263D;
    font-size: 29px;
    font-weight: 700;
    margin-top: 12px;
}

.gold-text {
    color: #B18A45;
}

div[data-testid="stForm"] {
    background: white;
    border: 1px solid #E7E3D9;
    padding: 25px;
    border-radius: 16px;
}

.stButton > button,
.stFormSubmitButton > button {
    background: #B18A45;
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 600;
    padding: 10px 22px;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #947039;
    color: white;
    border: none;
}

div[data-baseweb="tab-list"] {
    gap: 12px;
}

div[data-baseweb="tab"] {
    color: #17263D;
    font-weight: 600;
}

div[data-baseweb="tab-highlight"] {
    background-color: #B18A45;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD DATA
# ==================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"


@st.cache_data
def load_housing_data():
    file_path = DATA_DIR / "india_housing_prices.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Housing dataset not found at: {file_path}"
        )

    df = pd.read_csv(file_path)

    return df


@st.cache_data
def load_ml_data():
    file_path = DATA_DIR / "merged_houseprice_regression_classification.xlsx"

    if not file_path.exists():
        raise FileNotFoundError(
            f"ML dataset not found at: {file_path}"
        )

    return pd.read_excel(
        file_path,
        sheet_name="Regression+Class"
    )


@st.cache_resource
def load_models():
    classifier_path = MODEL_DIR / "classification_model.pkl"
    regressor_path = MODEL_DIR / "regression_model.pkl"

    if not classifier_path.exists() or not regressor_path.exists():
        raise FileNotFoundError(
            "One or both trained model files are missing from models/."
        )

    classifier = joblib.load(classifier_path)
    regressor = joblib.load(regressor_path)

    return classifier, regressor


housing_df = load_housing_data()
# Clean housing data for dashboard analysis
housing_df = housing_df.dropna().copy()

# Recalculate property age using the current year
housing_df["Age_of_Property"] = (
    2026 - housing_df["Year_Built"]
)

# Remove invalid property records
housing_df = housing_df[
    (housing_df["Price_in_Lakhs"] > 0) &
    (housing_df["Size_in_SqFt"] > 0) &
    (housing_df["BHK"] > 0)
].copy()

ml_df = load_ml_data()

classification_model, regression_model = load_models()


# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="hero">
    <h1>EstateIQ</h1>
    <p style="font-size:18px;">
        Real Estate Investment Advisor
    </p>
    <p>
        Explore property trends. Evaluate investment potential.
        Estimate future property value with machine learning.
    </p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# NAVIGATION
# ==================================================

explorer_tab, predictor_tab, about_tab = st.tabs([
    "🏘️  Property Explorer",
    "📈  Investment Predictor",
    "ℹ️  About"
])

# ==================================================
# PROPERTY EXPLORER
# ==================================================

with explorer_tab:

    st.markdown(
        '<div class="section-heading">Indian Housing Market</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Explore property records through location and property filters."
    )

    data = housing_df.dropna().copy()

    # Filters
    st.markdown("### Refine Your Search")

    c1, c2, c3 = st.columns(3)

    with c1:
        state = st.selectbox(
            "State",
            ["All"] + sorted(data["State"].unique().tolist())
        )

    with c2:
        property_type = st.selectbox(
            "Property Type",
            ["All"] + sorted(data["Property_Type"].unique().tolist())
        )

    with c3:
        bhk = st.selectbox(
            "Bedrooms",
            ["All"] + sorted(data["BHK"].unique().tolist())
        )

    filtered = data.copy()

    if state != "All":
        filtered = filtered[filtered["State"] == state]

    if property_type != "All":
        filtered = filtered[
            filtered["Property_Type"] == property_type
        ]

    if bhk != "All":
        filtered = filtered[filtered["BHK"] == bhk]

    st.markdown("---")
    st.markdown("### Market Overview")

    # Metrics
    m1, m2, m3, m4 = st.columns(4)

    count = len(filtered)

    avg_price = (
        filtered["Price_in_Lakhs"].mean() if count else 0
    )

    avg_size = (
        filtered["Size_in_SqFt"].mean() if count else 0
    )

    avg_bhk = (
        filtered["BHK"].mean() if count else 0
    )

    metrics = [
        ("Properties Found", f"{count:,}"),
        ("Average Price", f"₹{avg_price:.2f} L" if count else "N/A"),
        ("Average Area", f"{avg_size:,.0f} SqFt" if count else "N/A"),
        ("Average BHK", f"{avg_bhk:.1f}" if count else "N/A")
    ]

    for col, (label, value) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    if not filtered.empty:

        left, right = st.columns(2)

        with left:
            st.markdown("### Average Price by Property Type")

            price_type = (
                filtered.groupby("Property_Type")["Price_in_Lakhs"]
                .mean()
                .reset_index()
            )

            fig = px.bar(
                price_type,
                x="Property_Type",
                y="Price_in_Lakhs",
                color="Property_Type",
                color_discrete_sequence=["#B18A45", "#263D5A", "#8C9BAE"],
                labels={
                    "Property_Type": "Property Type",
                    "Price_in_Lakhs": "Price (₹ Lakhs)"
                }
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=False,
                margin=dict(l=15, r=15, t=25, b=15)
            )

            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("### Property Distribution by State")

            state_count = (
                filtered["State"]
                .value_counts()
                .head(10)
                .reset_index()
            )

            state_count.columns = ["State", "Properties"]

            fig = px.bar(
                state_count,
                x="Properties",
                y="State",
                orientation="h",
                color_discrete_sequence=["#B18A45"]
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=15, r=15, t=25, b=15)
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Property Price Distribution")

        fig = px.histogram(
            filtered,
            x="Price_in_Lakhs",
            nbins=30,
            color_discrete_sequence=["#263D5A"],
            labels={"Price_in_Lakhs": "Price (₹ Lakhs)"}
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=15, r=15, t=25, b=15)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Property Listings")

        st.dataframe(
            filtered.head(100),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Showing up to 100 records matching your selected filters."
        )

    else:
        st.warning("No properties found for these filters.")

# ==================================================
# INVESTMENT PREDICTOR
# ==================================================

with predictor_tab:

    st.markdown(
        '<div class="section-heading">Investment Intelligence</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Enter property details to estimate its future price category "
        "and five-year value."
    )

    st.info(
        "This prediction uses the separate 500-record ML training dataset."
    )

    feature_columns = [
        "Area_sqft",
        "Bedrooms",
        "Bathrooms",
        "Floors",
        "Year_Built",
        "Location_Score",
        "School_Score",
        "Current_Price"
    ]

    labels = {
        "Area_sqft": "Property Area (SqFt)",
        "Bedrooms": "Number of Bedrooms",
        "Bathrooms": "Number of Bathrooms",
        "Floors": "Number of Floors",
        "Year_Built": "Year Built",
        "Location_Score": "Location Score",
        "School_Score": "School Score",
        "Current_Price": "Current Price (₹)"
    }

    st.markdown("### Property Information")

    with st.form("investment_form"):

        col1, col2 = st.columns(2)

        input_values = {}

        for index, feature in enumerate(feature_columns):

            column = col1 if index % 2 == 0 else col2

            with column:

                minimum = float(ml_df[feature].min())
                maximum = float(ml_df[feature].max())
                default = float(ml_df[feature].median())

                integer_fields = [
                    "Area_sqft",
                    "Bedrooms",
                    "Bathrooms",
                    "Floors",
                    "Year_Built"
                ]

                is_integer = feature in integer_fields

                input_values[feature] = st.number_input(
                    labels[feature],
                    min_value=int(minimum) if is_integer else minimum,
                    max_value=int(maximum) if is_integer else maximum,
                    value=int(default) if is_integer else default,
                    step=1 if is_integer else 0.1
                )

        submitted = st.form_submit_button(
            "✦  Analyze Investment",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame(
            [input_values],
            columns=feature_columns
        )

        predicted_category = classification_model.predict(
            input_df
        )[0]

        predicted_price = regression_model.predict(
            input_df
        )[0]

        st.markdown("---")
        st.markdown("### Your Investment Analysis")

        r1, r2 = st.columns(2)

        with r1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">
                    Predicted Price Category
                </div>
                <div class="result-value gold-text">
                    {predicted_category}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">
                    Estimated Value After Five Years
                </div>
                <div class="result-value">
                    ₹{predicted_price:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.success("Your property analysis is ready!")

        st.caption(
            "The category represents the dataset's future-price band. "
            "Predictions are illustrative and are not guaranteed returns "
            "or financial advice."
        )

# ==================================================
# ABOUT
# ==================================================

with about_tab:

    st.markdown(
        '<div class="section-heading">About EstateIQ</div>',
        unsafe_allow_html=True
    )

    st.write(
        "EstateIQ is a machine-learning-based real estate investment "
        "analysis application."
    )

    st.markdown("### Technology Stack")

    technologies = [
        ("Python", "Core programming language"),
        ("Pandas & NumPy", "Data handling and preprocessing"),
        ("Scikit-learn", "Classification and regression models"),
        ("Plotly", "Interactive data visualizations"),
        ("Streamlit", "Interactive web application"),
        ("MLflow", "Experiment tracking")
    ]

    for technology, purpose in technologies:
        st.markdown(f"**{technology}** — {purpose}")

    st.markdown("### Machine Learning Models")

    st.write(
        "- Classification: Gradient Boosting Classifier\n"
        "- Regression: Linear Regression"
    )

    st.markdown("### Dataset Information")

    st.write(
        "The application uses two separate datasets. The Indian housing "
        "dataset powers property exploration, while the 500-record "
        "regression/classification dataset powers ML predictions."
    )

    st.warning(
        "The prediction dataset contains formula-like future price "
        "patterns. Model results should be interpreted as dataset-specific "
        "experiments, not validated real-world market forecasts."
    )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#778397; padding:12px;">
        <b>EstateIQ</b> · Real Estate Investment Advisor<br>
        Machine Learning Project · Built with Python and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

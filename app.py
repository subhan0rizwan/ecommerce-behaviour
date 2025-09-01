import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.preprocessing import LabelEncoder

# ==============================
# Configuration
# ==============================
APP_TITLE = "E-commerce Churn Prediction Dashboard"
APP_SUBTITLE = "Built by Subhan | Advanced Churn Analysis | Updated: Sep 01, 2025"

PRIMARY_COLOR = "#DE9485"     # warm coral-ish
SECONDARY_COLOR = "#14BBCA"   # teal accent
WARM_BG_COLOR = "#FFF7F0"
DARK_BG_COLOR = "#121212"

# ==============================
# Custom CSS (light + dark)
# ==============================
st.markdown(f"""
<style>
    /* 🌞 Light Mode */
    @media (prefers-color-scheme: light) {{
        .main {{
            background-color: {WARM_BG_COLOR};
            color: #000000;
            font-family: 'Segoe UI', sans-serif;
        }}
        h1, h2, h3, h4 {{ color: {PRIMARY_COLOR}; }}
        section[data-testid="stSidebar"] {{
            background-color: #ffffff;
            border-right: 2px solid {PRIMARY_COLOR};
        }}
        .stButton>button {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.6em 1.2em;
            font-weight: bold;
        }}
        .stButton>button:hover {{ background-color: {SECONDARY_COLOR}; }}
    }}
    /* 🌙 Dark Mode */
    @media (prefers-color-scheme: dark) {{
        .main {{
            background-color: #0d0d0d;
            color: #f2f2f2;
            font-family: 'Courier New', monospace;
        }}
        h1, h2, h3, h4 {{
            color: #ff00ff;
            text-shadow: 0px 0px 10px #ff00ff, 0px 0px 20px #ff33cc;
        }}
        section[data-testid="stSidebar"] {{
            background-color: #1a1a1a;
            border-right: 2px solid #ff00ff;
        }}
        .stButton>button {{
            background: linear-gradient(90deg, #ff00ff, #00ffff);
            color: black;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            text-shadow: 0px 0px 5px #00ffff;
            box-shadow: 0px 0px 10px #ff00ff, 0px 0px 20px #00ffff;
        }}
        .stButton>button:hover {{
            background: linear-gradient(90deg, #00ffff, #ff00ff);
            color: white;
            box-shadow: 0px 0px 15px #00ffff, 0px 0px 25px #ff00ff;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# ==============================
# App Title
# ==============================
st.markdown(f"<h1 style='text-align: center;'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center;'>{APP_SUBTITLE}</p>", unsafe_allow_html=True)

# ==============================
# Data & Model Loading
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv("churn_features.csv")

@st.cache_resource
def load_model():
    bundle = joblib.load("churn_lgb_bundle.pkl")
    return bundle["model"], bundle["scaler"]

try:
    df = load_data()
    model, scaler = load_model()
except Exception as e:
    st.error(f"⚠️ Failed to load data/model: {str(e)}")
    st.stop()

# Pre-fit encoders for categorical features
ENCODERS = {}
for col in ['most_frequent_category', 'most_frequent_country', 'most_frequent_brand']:
    if col in df.columns:
        le = LabelEncoder()
        le.fit(df[col].astype(str).dropna())
        ENCODERS[col] = le

# ==============================
# Utility Functions
# ==============================
def make_chart(fig):
    with st.container():
        st.plotly_chart(fig, use_container_width=True)

def predict_churn(input_data: pd.DataFrame):
    """Preprocess, align features, and predict churn."""
    for col, le in ENCODERS.items():
        if col in input_data:
            input_data[col] = le.transform(input_data[col].astype(str))

    expected_features = [c for c in df.drop(columns=['user_id', 'churn'], errors='ignore').columns]
    input_data = input_data.reindex(columns=expected_features, fill_value=0)

    input_scaled = scaler.transform(input_data)
    proba = model.predict_proba(input_scaled)[0]
    return (int(proba[1] > 0.5), proba[1])

# ==============================
# Sidebar
# ==============================
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", [
    "Dashboard", "Data Overview", "Feature Importance", 
    "Churn Prediction by Category", "Churn Prediction by Country", "CLV Analysis"
])

countries = st.sidebar.multiselect(
    "Filter by Country", 
    df['most_frequent_country'].unique(), 
    default=df['most_frequent_country'].unique()
)
df_filtered = df[df['most_frequent_country'].isin(countries)]

# ==============================
# Pages
# ==============================
if page == "Dashboard":
    st.header("Churn Analytics Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Churn Rate", f"{df_filtered['churn'].mean()*100:.2f}%")
    col2.metric("Avg. Purchases/User", f"{df_filtered['total_purchases'].mean():.2f}")
    col3.metric("Top Category", df_filtered['most_frequent_category'].mode().iat[0])

    st.subheader("Churn by Brand")
    churn_by_brand = df_filtered.groupby('most_frequent_brand')['churn'].mean().reset_index()
    fig = px.bar(
        churn_by_brand, x='most_frequent_brand', y='churn',
        title="Churn Rate by Brand", color='churn',
        color_continuous_scale="Inferno", template="plotly_dark"
    )
    make_chart(fig)
    
elif page == "Data Overview":
    st.header("Data Overview")
    st.dataframe(df_filtered.head())

    st.subheader("Churn Distribution")
    fig = px.histogram(
        df_filtered, x='churn', title="Churn Distribution",
        color='churn', color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR],
        template="plotly_dark"
    )
    make_chart(fig)

elif page == "Feature Importance":
    st.header("Feature Importance")
    try:
        # Try to extract feature names from file (saved during training)
        if hasattr(model, "feature_importances_"):
            try:
                # load feature names saved during training
                with open("feature_names.txt", "r") as f:
                    feature_names = [line.strip() for line in f.readlines()]
            except:
                # fallback if no file exists
                feature_names = [f"Feature {i}" for i in range(len(model.feature_importances_))]

            feature_importances = model.feature_importances_

            fig = px.bar(
                x=feature_importances,
                y=feature_names,
                orientation="h",
                title="LightGBM Feature Importance",
                color=feature_importances,
                color_continuous_scale="Blues",
                template="plotly_dark"
            )
            make_chart(fig)

        # Always show SHAP summary from XGBoost
        st.image("shap_summary.png", caption="SHAP Summary Plot (XGBoost)", width=700)

    except Exception as e:
        st.warning(f"⚠️ Feature importance not available: {e}")

elif page == "Churn Prediction by Category":
    st.header("Predict Churn by Category")
    with st.form("category_form"):
        selected_category = st.selectbox("Most Frequent Category", df_filtered['most_frequent_category'].unique())
        avg_purchases = st.slider("Total Purchases", 0, 20, 5)
        last_purchase = st.slider("Days Since Last Purchase", 0, 90, 30)
        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = pd.DataFrame({
                "most_frequent_category": [selected_category],
                "total_purchases": [avg_purchases],
                "time_since_last_purchase": [last_purchase],
                "most_frequent_country": [df_filtered['most_frequent_country'].mode()[0]],
                "most_frequent_brand": [df_filtered['most_frequent_brand'].mode()[0]],
            })
            pred, prob = predict_churn(input_data)
            st.success(f"Prediction: {'Churn' if pred else 'No Churn'}")
            st.info(f"Churn Probability: {prob:.2%}")

elif page == "Churn Prediction by Country":
    st.header("Predict Churn by Country")
    with st.form("country_form"):
        selected_country = st.selectbox("Most Frequent Country", df_filtered['most_frequent_country'].unique())
        avg_purchases = st.slider("Total Purchases", 0, 20, 5)
        last_purchase = st.slider("Days Since Last Purchase", 0, 90, 30)
        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = pd.DataFrame({
                "most_frequent_country": [selected_country],
                "total_purchases": [avg_purchases],
                "time_since_last_purchase": [last_purchase],
                "most_frequent_category": [df_filtered['most_frequent_category'].mode()[0]],
                "most_frequent_brand": [df_filtered['most_frequent_brand'].mode()[0]],
            })
            pred, prob = predict_churn(input_data)
            st.success(f"Prediction: {'Churn' if pred else 'No Churn'}")
            st.info(f"Churn Probability: {prob:.2%}")

elif page == "CLV Analysis":
    st.header("Customer Lifetime Value (CLV) Analysis")
    try:
        df["clv"] = df["avg_purchase_value"] * df["total_purchases"] * (1 - df["churn"].mean())
        clv_by_category = df.groupby("most_frequent_category")["clv"].mean().reset_index()
        fig = px.bar(
            clv_by_category, x="most_frequent_category", y="clv",
            title="Average CLV by Category", color="clv",
            color_continuous_scale="Blues", template="plotly_dark"
        )
        make_chart(fig)
    except Exception as e:
        st.error(f"⚠️ CLV analysis failed: {str(e)}")

# ==============================
# Download
# ==============================
st.download_button("⬇️ Download Filtered Data", df_filtered.to_csv(index=False), file_name="churn_data.csv")


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💼",
    layout="wide"
)

# ================= INDIAN CURRENCY FORMAT =================
def format_inr(amount):
    s = str(int(round(amount)))
    if len(s) <= 3:
        return s
    last3 = s[-3:]
    rest = s[:-3]
    rest = ",".join(
        [rest[max(i-2, 0):i] for i in range(len(rest), 0, -2)][::-1]
    )
    return rest + "," + last3

# ================= GLOBAL DARK UI =================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e5e7eb;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
}

@keyframes glow {
    0% { text-shadow: 0 0 8px #6366f1; }
    50% { text-shadow: 0 0 18px #22d3ee; }
    100% { text-shadow: 0 0 8px #6366f1; }
}

.glow-title {
    font-size: 3rem;
    font-weight: 800;
    animation: glow 3s infinite ease-in-out;
}

div[data-testid="metric-container"] {
    background: linear-gradient(145deg, #020617, #0f172a);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #1e293b;
}

h2, h3, h4, p, label {
    color: #e5e7eb !important;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL & DATA =================
@st.cache_resource
def load_model():
    data = joblib.load("salary_prediction_model.pkl")
    return data["model"]

@st.cache_data
def load_data():
    return pd.read_csv("hrdataset.csv")

model = load_model()
df = load_data()

FEATURES = ["Age", "Bonus", "Months", "Education"]

# ================= SESSION STATE =================
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ================= TITLE =================
st.markdown(
    "<div class='glow-title'>Salary Prediction System</div>",
    unsafe_allow_html=True
)
st.caption("Predict salary using demographic, financial, and business attributes")

# ================= SIDEBAR =================
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("", ["Overview", "Insights", "Salary Prediction"])

# ================= OVERVIEW =================
if page == "Overview":
    st.subheader("Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Records", df.shape[0])
    c2.metric("Features", len(FEATURES))
    c3.metric("Model", "Random Forest")

    st.markdown("""
    This system predicts employee salary using:
    - Demographic attributes  
    - Experience and incentives  
    - Educational background  

    Built for **stability**, **accuracy**, and **real-world deployment**.
    """)

# ================= INSIGHTS =================
elif page == "Insights":
    st.subheader("Insights")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df["Salary"], bins=30, kde=True, ax=ax, color="#6366f1")
        ax.set_facecolor("#020617")
        fig.patch.set_facecolor("#020617")
        ax.set_title("Salary Distribution", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=df["Age"],
            y=df["Salary"],
            ax=ax,
            color="#22d3ee",
            s=40
        )
        ax.set_facecolor("#020617")
        fig.patch.set_facecolor("#020617")
        ax.set_title("Age vs Salary", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig)

    st.markdown("### Feature Correlation")
    corr = df.select_dtypes(include=np.number).corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    ax.set_facecolor("#020617")
    fig.patch.set_facecolor("#020617")
    ax.tick_params(colors="white")
    ax.set_title("Correlation Matrix", color="white")
    st.pyplot(fig)

# ================= SALARY PREDICTION =================
elif page == "Salary Prediction":
    st.subheader("Predict Salary")

    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    bonus = st.number_input("Bonus", value=5000.0)
    months = st.number_input("Months of Experience", value=12)

    # 🔒 DROP NaN VALUES FROM UI
    education = st.selectbox(
        "Education",
        df["Education"].dropna().unique()
    )

    if st.button("Predict Salary"):
        # 🔒 FORCE EXACT FEATURE ORDER
        input_df = pd.DataFrame(
            [[age, bonus, months, education]],
            columns=FEATURES
        )

        st.session_state.prediction = model.predict(input_df)[0]

    if st.session_state.prediction is not None:
        formatted_salary = format_inr(st.session_state.prediction)

        st.markdown(
            f"""
            <div style="
                font-size:2.4rem;
                font-weight:800;
                color:#22d3ee;
                text-shadow:0 0 20px #22d3ee;
            ">
            ₹ {formatted_salary}
            </div>
            """,
            unsafe_allow_html=True
        )

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#9ca3af;'>Built by <b>Aryan Pankaj Kumar</b></div>",
    unsafe_allow_html=True
)

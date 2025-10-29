import os
import json
import requests
import PyPDF2
import pandas as pd
import streamlit as st

# --------------------------
# Page Setup
# --------------------------
st.set_page_config(page_title="AI Skills Radar", layout="wide")

# --------------------------
# Load Gemini API Key Securely
# --------------------------
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    st.error("⚠️ Gemini API key not configured. Please check your .streamlit/secrets.toml file.")
    st.stop()

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
)

# --------------------------
# Custom CSS and JavaScript
# --------------------------
st.markdown(
    """
    <style>
    /* Background and fonts */
    body, .stApp {
        background-color: #f3e8ff;
        font-family: 'Poppins', sans-serif;
    }
    /* Navbar */
    .navbar {
        background-color: #7b2cbf;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        border-radius: 0 0 20px 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .navbar span {
        color: #d0bfff;
    }
    /* Upload boxes and buttons */
    .stButton>button {
        background-color: #9d4edd;
        color: white;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #7b2cbf;
        color: #fff;
        transform: scale(1.03);
        transition: all 0.2s ease-in-out;
    }
    .result-box {
        background-color: #ffffffb8;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# Navbar
# --------------------------
st.markdown(
    """
    <div class="navbar">
        🌐 <span>AI Skills Radar</span> | Team Skill Gap Analyzer
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# Helper Functions
# --------------------------
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


def analyze_skill_gaps(jd_text, team_data):
    """Call Gemini API to compare JD vs team skills"""
    prompt = f"""
    You are an HR talent intelligence assistant.
    Analyze the following:
    1️⃣ Job Description (JD):
    {jd_text}

    2️⃣ Team Member Skills:
    {team_data.to_string(index=False)}

    Compare and identify:
    - Top skill matches
    - Missing/gap skills
    - Suggested internal mentors or training areas
    - Predict future skill needs (next 6 months)
    Output in clear, structured format with recommendations.
    """

    response = requests.post(
        GEMINI_API_URL,
        headers={"Content-Type": "application/json"},
        params={"key": GEMINI_API_KEY},
        json={"contents": [{"parts": [{"text": prompt}]}]},
    )

    try:
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Error parsing response: {e}\n\nRaw: {data}"


# --------------------------
# Main Interface
# --------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload Job Description (JD)")
    jd_file = st.file_uploader("Upload a JD (PDF or text)", type=["pdf", "txt"])

with col2:
    st.subheader("👥 Upload Team Profiles (CSV)")
    team_file = st.file_uploader("Upload Team Skills File", type=["csv"])

# --------------------------
# Run Analysis
# --------------------------
if jd_file and team_file:
    if st.button("🚀 Analyze Skill Gaps"):
        with st.spinner("Analyzing skills... please wait..."):
            if jd_file.name.endswith(".pdf"):
                jd_text = extract_text_from_pdf(jd_file)
            else:
                jd_text = jd_file.read().decode("utf-8")

            team_df = pd.read_csv(team_file)
            result = analyze_skill_gaps(jd_text, team_df)

        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown("### 📊 AI-Generated Skill Gap Insights")
        st.write(result)
        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Dashboard Preview
# --------------------------
st.markdown("### 📈 Team Readiness Dashboard (Preview)")
sample_data = {
    "Skill Category": ["AI / Data", "Leadership", "Cloud", "Compliance"],
    "Current Level (%)": [65, 80, 50, 75],
    "Target 2025 (%)": [85, 90, 75, 90],
}
df = pd.DataFrame(sample_data)
st.bar_chart(df.set_index("Skill Category"))

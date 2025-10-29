import os
import re
import json
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from io import StringIO
from PyPDF2 import PdfReader

# ---------------------------
# Streamlit Config
# ---------------------------
st.set_page_config(page_title="AI Skills Radar", layout="wide")

# ---------------------------
# Initialize Session State for Page Navigation
# ---------------------------
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Home'

# ---------------------------
# Load Gemini API Key
# ---------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

if not GEMINI_API_KEY:
    st.error("⚠️ Gemini API key not found. Please set it in Streamlit Secrets or environment variables.")

# ---------------------------
# Enhanced CSS with Black & White Theme
# ---------------------------
st.markdown("""
<style>
/* Import modern font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Reset and base styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
    color: #1a1a1a;
    font-family: 'Inter', sans-serif;
    overflow-x: hidden;
}

/* Fixed top navbar */
.top-navbar {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    z-index: 9999;
    background-color: #1a1a1a;
    padding: 0;
    margin: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
}

.nav-left {
    display: flex;
    align-items: center;
    padding: 16px 40px;
}

.nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.nav-logo {
    font-size: 24px;
}

.nav-title {
    font-weight: 700;
    font-size: 18px;
    color: white;
    letter-spacing: 0.5px;
}

.nav-links {
    list-style-type: none;
    margin: 0;
    padding: 0;
    display: flex;
    align-items: center;
    gap: 12px;
    padding-right: 40px;
}

.nav-link {
    display: block;
    color: white;
    padding: 10px 24px;
    text-decoration: none;
    font-weight: 500;
    font-size: 15px;
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    transition: all 0.3s ease;
    cursor: pointer;
}

.nav-link:hover {
    background-color: rgba(255, 255, 255, 0.2);
    transform: translateY(-2px);
}

.nav-link.active {
    background-color: rgba(255, 255, 255, 0.25);
}

/* Hide Streamlit default elements and navigation buttons */
.stElementContainer, .stMarkdown {
    margin: 0 !important;
    padding: 0 !important;
}

/* Hide all Streamlit elements */
header[data-testid="stHeader"] {
    display: none !important;
}

.stApp > header {
    display: none !important;
}

/* Hide navigation buttons completely */
.element-container:has(button) {
    display: none !important;
}

div[data-testid="column"] {
    display: none !important;
}

/* Main container */
.main-container {
    padding-top: 70px;
    padding-bottom: 60px;
    max-width: 1400px;
    margin: 0 auto;
    padding-left: 40px;
    padding-right: 40px;
    min-height: calc(100vh - 250px);
}

/* Section cards with modern design */
.section-card {
    background: white;
    border-radius: 20px;
    padding: 40px;
    margin: 15px 0;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
    border: 1px solid rgba(0, 0, 0, 0.05);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.section-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #000000 0%, #404040 100%);
    transform: scaleX(0);
    transition: transform 0.4s ease;
}

.section-card:hover::before {
    transform: scaleX(1);
}

.section-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
    border-color: rgba(0, 0, 0, 0.1);
}

/* Section headers */
.section-header {
    font-size: 28px;
    font-weight: 700;
    color: #000;
    margin-bottom: 24px;
    letter-spacing: -0.5px;
    position: relative;
    display: inline-block;
}

.section-header::after {
    content: '';
    position: absolute;
    bottom: -8px;
    left: 0;
    width: 60px;
    height: 3px;
    background: #000;
    transition: width 0.3s ease;
}

.section-card:hover .section-header::after {
    width: 100%;
}

/* Enhanced buttons */
.stButton>button {
    background: linear-gradient(135deg, #000000 0%, #2d2d2d 100%) !important;
    color: #fff !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton>button::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.2);
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
}

.stButton>button:hover::before {
    width: 300px;
    height: 300px;
}

.stButton>button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3) !important;
}

.stButton>button:active {
    transform: translateY(-1px) !important;
}

/* Input fields */
.stTextInput>div>div>input,
.stTextArea>div>div>textarea {
    border: 2px solid #e0e0e0 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    transition: all 0.3s ease !important;
    background: #fafafa !important;
}

.stTextInput>div>div>input:focus,
.stTextArea>div>div>textarea:focus {
    border-color: #000 !important;
    box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.1) !important;
    background: white !important;
}

/* File uploader */
.stFileUploader>div>div {
    border: 2px dashed #d0d0d0 !important;
    border-radius: 16px !important;
    padding: 30px !important;
    background: #fafafa !important;
    transition: all 0.3s ease !important;
}

.stFileUploader>div>div:hover {
    border-color: #000 !important;
    background: white !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08) !important;
}

/* Metric cards */
.stMetric {
    background: linear-gradient(135deg, #f8f8f8 0%, #ffffff 100%);
    padding: 20px;
    border-radius: 16px;
    border: 2px solid #e0e0e0;
    transition: all 0.3s ease;
}

.stMetric:hover {
    border-color: #000;
    transform: scale(1.02);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
}

/* Tables */
.stTable {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

/* Dataframe */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e0e0e0;
}

/* Footer */
.app-footer {
    background: #1a1a1a;
    color: #e0e0e0;
    position: relative;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 30px 40px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    margin-top: 40px;
}

.footer-content {
    max-width: 1400px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: 2fr 1fr 1fr;
    gap: 40px;
    text-align: left;
}

.footer-section h3 {
    color: white;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 12px;
}

.footer-section p {
    font-size: 14px;
    line-height: 1.6;
    color: #b0b0b0;
    margin: 6px 0;
}

.footer-section a {
    color: #b0b0b0;
    text-decoration: none;
    transition: color 0.3s ease;
}

.footer-section a:hover {
    color: white;
}

.footer-bottom {
    text-align: center;
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    font-size: 13px;
    color: #888;
}

/* Auth form styling */
.auth-form {
    max-width: 500px;
    margin: 0 auto;
    background: white;
    padding: 40px;
    border-radius: 16px;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.08);
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.section-card {
    animation: fadeInUp 0.6s ease both;
}

/* Smooth scroll */
html {
    scroll-behavior: smooth;
}

/* Loading spinner */
.stSpinner>div {
    border-color: #000 transparent transparent transparent !important;
}

/* Success/Error messages */
.stSuccess {
    background-color: #f0f0f0 !important;
    color: #000 !important;
    border-left: 4px solid #000 !important;
}

.stError {
    background-color: #fff5f5 !important;
    color: #1a1a1a !important;
    border-left: 4px solid #ff4444 !important;
}

.stWarning {
    background-color: #fffef5 !important;
    color: #1a1a1a !important;
    border-left: 4px solid #ffa500 !important;
}

/* Hero section for home page */
.hero-section {
    background: linear-gradient(135deg, #000000 0%, #2d2d2d 100%);
    color: white;
    padding: 60px 40px;
    border-radius: 20px;
    text-align: center;
    margin: 15px 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.hero-title {
    font-size: 48px;
    font-weight: 700;
    margin-bottom: 20px;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 20px;
    color: #e0e0e0;
    margin-bottom: 30px;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 24px;
    margin-top: 30px;
}

.feature-card {
    background: white;
    padding: 30px;
    border-radius: 16px;
    text-align: center;
    transition: all 0.3s ease;
    border: 2px solid #f0f0f0;
}

.feature-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15);
    border-color: #000;
}

.feature-icon {
    font-size: 48px;
    margin-bottom: 16px;
}

.feature-title {
    font-size: 20px;
    font-weight: 600;
    color: #000;
    margin-bottom: 12px;
}

.feature-desc {
    font-size: 14px;
    color: #666;
    line-height: 1.6;
}

/* Responsive design */
@media (max-width: 768px) {
    .top-navbar {
        flex-direction: column;
        align-items: flex-start;
    }
    
    .nav-left {
        width: 100%;
        padding: 12px 20px;
    }
    
    .nav-links {
        width: 100%;
        flex-direction: column;
        padding: 0 20px 12px 20px;
        gap: 8px;
    }
    
    .nav-link {
        width: 100%;
        padding: 12px 20px;
        text-align: center;
    }
    
    .main-container {
        padding-left: 20px;
        padding-right: 20px;
        padding-top: 160px;
    }
    
    .section-card {
        padding: 24px;
        margin: 20px 0;
    }
    
    .section-header {
        font-size: 22px;
    }
    
    .hero-title {
        font-size: 32px;
    }
    
    .hero-subtitle {
        font-size: 16px;
    }
    
    .footer-content {
        grid-template-columns: 1fr;
        gap: 24px;
    }
    
    .app-footer {
        padding: 20px;
    }
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
}

::-webkit-scrollbar-thumb {
    background: #000;
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: #333;
}
</style>

<script>
/* Navbar scroll effect - removed since we want solid black */
/* Removed scroll effect to keep navbar consistently black */
</script>
""", unsafe_allow_html=True)

# ---------------------------
# Navigation Functions
# ---------------------------
def set_page(page_name):
    st.session_state['current_page'] = page_name
    st.rerun()

# ---------------------------
# Render Navigation Bar
# ---------------------------
current_page = st.session_state['current_page']

st.markdown(f"""
<div class="top-navbar">
  <div class="nav-left">
    <div class="nav-brand">
      <span class="nav-logo">📊</span>
      <span class="nav-title">AI Skills Radar</span>
    </div>
  </div>
  <ul class="nav-links">
    <li><a class="nav-link {'active' if current_page == 'Home' else ''}" id="nav-home">🏠 Home</a></li>
    <li><a class="nav-link {'active' if current_page == 'Account' else ''}" id="nav-account">🔐 Account</a></li>
    <li><a class="nav-link {'active' if current_page == 'Upload & Analyze' else ''}" id="nav-upload">📁 Upload & Analyze</a></li>
  </ul>
</div>
""", unsafe_allow_html=True)

# JavaScript to handle navigation clicks
st.markdown("""
<script>
document.getElementById('nav-home').onclick = function() {
    window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'Home'}, '*');
};
document.getElementById('nav-account').onclick = function() {
    window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'Account'}, '*');
};
document.getElementById('nav-upload').onclick = function() {
    window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'Upload & Analyze'}, '*');
};
</script>
""", unsafe_allow_html=True)

# Navigation buttons (hidden but functional) - Keep these for functionality
if 'button_clicked' not in st.session_state:
    st.session_state['button_clicked'] = False

# Create a hidden container for navigation
nav_container = st.container()
with nav_container:
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("home", key="nav_home_btn", type="secondary"):
            set_page('Home')
    with col2:
        if st.button("account", key="nav_account_btn", type="secondary"):
            set_page('Account')
    with col3:
        if st.button("upload", key="nav_upload_btn", type="secondary"):
            set_page('Upload & Analyze')

# ---------------------------
# Skill Extract & Analyzer Logic
# ---------------------------
COMMON_SKILLS = [
    'python','sql','excel','data analysis','communication','project management',
    'leadership','product management','machine learning','nlp','aws','gcp','react',
    'java','c#','sales','negotiation','recruiting','interviewing','coaching','training'
]

def extract_skills_from_text(text):
    text_lower = (text or "").lower()
    found = [s for s in COMMON_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', text_lower)]
    if not found:
        tokens = re.findall(r"[a-zA-Z]{4,}", text_lower)
        found = tokens[:8]
    return sorted(set(found))

def aggregate_team_skills(df):
    counts = {}
    if df is None:
        return counts
    if 'skills' in [c.lower() for c in df.columns]:
        col = next(c for c in df.columns if c.lower() == 'skills')
        for row in df[col].dropna():
            for s in [x.strip().lower() for x in re.split('[,;|/\\\\n]', str(row)) if x.strip()]:
                counts[s] = counts.get(s, 0) + 1
    else:
        for _, r in df.iterrows():
            combined = ' '.join([str(x) for x in r.values if pd.notna(x)])
            for s in extract_skills_from_text(combined):
                counts[s] = counts.get(s, 0) + 1
    return counts

def compute_skill_match(jd_skills, team_skill_counts):
    jd_set = set(jd_skills)
    team_set = set(team_skill_counts.keys())
    matched = jd_set & team_set
    missing = jd_set - team_set
    score = int(100 * len(matched) / max(1, len(jd_set)))
    missing_detail = [{'skill': s, 'team_count': team_skill_counts.get(s, 0)} for s in sorted(missing)]
    return score, sorted(matched), missing_detail

def radar_chart(skills, values, title='Team Skills Radar'):
    N = len(skills)
    if N == 0:
        return None
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    vals = list(values)
    vals += vals[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, vals, linewidth=2, color='black')
    ax.fill(angles, vals, alpha=0.25, color='gray')
    ax.set_thetagrids(np.degrees(angles[:-1]), skills)
    ax.set_ylim(0,100)
    ax.set_title(title, fontsize=14, fontweight='bold')
    return fig

def call_gemini(prompt, system_prompt="You are an AI HR analyst generating concise workforce insights.", max_tokens=400):
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": f"{system_prompt}\n\n{prompt}"}]}]}
    params = {"key": GEMINI_API_KEY}
    resp = requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload, timeout=30)
    data = resp.json()
    try:
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error parsing response: {e}\n\nRaw: {data}"

# ---------------------------
# Main Content Container
# ---------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ---------------------------
# PAGE: HOME
# ---------------------------
if current_page == 'Home':
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">Welcome to AI Skills Radar</h1>
        <p class="hero-subtitle">Your intelligent workforce analytics platform powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <h3 class="feature-title">Align Skills</h3>
            <p class="feature-desc">Match job requirements with current team capabilities in real-time</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3 class="feature-title">Visualize Gaps</h3>
            <p class="feature-desc">Identify skill gaps through interactive radar charts and analytics</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h3 class="feature-title">AI Insights</h3>
            <p class="feature-desc">Get actionable recommendations powered by Gemini AI technology</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <h3 class="feature-title">Plan Growth</h3>
            <p class="feature-desc">Develop targeted upskilling and hiring strategies for your team</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🚀 Getting Started</h2>', unsafe_allow_html=True)
    st.markdown("""
    ### How to use AI Skills Radar:
    
    1. **Create an Account** - Navigate to the Account page to sign up or log in
    2. **Upload Your Data** - Go to Upload & Analyze to upload:
       - Job Description (TXT or PDF format)
       - Team Profiles (CSV or Excel format)
    3. **Run Analysis** - Click the analyze button to get instant insights
    4. **Review Results** - Get match scores, skill gaps, visualizations, and AI recommendations
    
    ### Why Choose AI Skills Radar?
    
    - ⚡ **Fast & Efficient** - Get results in seconds
    - 🎯 **Accurate Matching** - Advanced algorithms for precise skill mapping
    - 📊 **Visual Insights** - Easy-to-understand charts and metrics
    - 🤖 **AI-Powered** - Smart recommendations for workforce development
    - 💼 **HR-Focused** - Built specifically for HR and L&D professionals
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# PAGE: ACCOUNT
# ---------------------------
elif current_page == 'Account':
    # Check if user is already logged in
    if 'user' in st.session_state:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">👤 User Profile</h2>', unsafe_allow_html=True)
        
        # Centered profile info
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <h3>Welcome, {st.session_state['user']}!</h3>
                <p><strong>Account Status:</strong> Active ✅</p>
                <p><strong>Access Level:</strong> Full Access</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
                del st.session_state['user']
                st.success("Logged out successfully!")
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Centered auth form
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="section-card auth-form">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-header" style="text-align: center;">🔐 Account</h2>', unsafe_allow_html=True)
            
            # Single form with tabs for Login/Signup
            tab1, tab2 = st.tabs(["🔓 Login", "✨ Sign Up"])
            
            with tab1:
                st.markdown("### Welcome Back!")
                user = st.text_input("Username", key="login_user", placeholder="Enter your username")
                pwd = st.text_input("Password", type="password", key="login_pwd", placeholder="Enter your password")
                if st.button("🔓 Login", key="login_btn", use_container_width=True):
                    if user == "admin" and pwd == "1234":
                        st.success("✅ Logged in successfully!")
                        st.session_state['user'] = user
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please try again.")
            
            with tab2:
                st.markdown("### Create New Account")
                new_user = st.text_input("Username", key="signup_user", placeholder="Choose a username")
                new_pwd = st.text_input("Password", type="password", key="signup_pwd", placeholder="Choose a password")
                confirm_pwd = st.text_input("Confirm Password", type="password", key="confirm_pwd", placeholder="Confirm your password")
                if st.button("✨ Create Account", key="signup_btn", use_container_width=True):
                    if new_user and new_pwd:
                        if new_pwd == confirm_pwd:
                            st.success(f"✅ Account '{new_user}' created successfully!")
                            st.session_state['user'] = new_user
                            st.balloons()
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Passwords do not match!")
                    else:
                        st.error("❌ Please provide both username and password.")
            
            st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# PAGE: UPLOAD & ANALYZE
# ---------------------------
elif current_page == 'Upload & Analyze':
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📁 Upload Files</h2>', unsafe_allow_html=True)
    
    # Upload subsection
    st.markdown("### 📤 Upload Your Data")
    col_jd, col_team = st.columns(2)
    
    with col_jd:
        st.markdown("**Job Description**")
        jd_file = st.file_uploader("Upload JD (TXT/PDF)", type=['txt','pdf'], key="jd_uploader", label_visibility="collapsed")
    
    with col_team:
        st.markdown("**Team Profiles**")
        team_file = st.file_uploader("Upload Team Data (CSV/Excel)", type=['csv','xlsx','xls'], key="team_uploader", label_visibility="collapsed")
    
    jd_text = ""
    
    if jd_file:
        if jd_file.type == "text/plain":
            jd_text = jd_file.getvalue().decode("utf-8")
        elif jd_file.type == "application/pdf":
            reader = PdfReader(jd_file)
            jd_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        st.text_area("📄 Job Description Preview (editable)", jd_text, height=200, key="jd_preview")
        st.session_state['jd_text'] = jd_text
    
    team_df = None
    if team_file:
        try:
            team_df = pd.read_csv(team_file) if team_file.name.endswith(".csv") else pd.read_excel(team_file)
            st.markdown("### 👥 Team Profiles Preview")
            st.dataframe(team_df.head(), use_container_width=True)
            st.session_state['team_df'] = team_df
        except Exception as e:
            st.error(f"❌ Failed to read team file: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis section
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🔍 Run Analysis</h2>', unsafe_allow_html=True)
    
    if st.button("🚀 Analyze Skills Gap", key="analyze_btn", use_container_width=True):
        # Get data from session state if not uploaded in current session
        if not jd_text and 'jd_text' in st.session_state:
            jd_text = st.session_state['jd_text']
        if not jd_text:
            st.warning("⚠️ Please provide a Job Description (upload or paste).")
        elif team_df is None and 'team_df' not in st.session_state:
            st.warning("⚠️ Please upload team profiles (CSV/Excel).")
        else:
            if 'team_df' in st.session_state and team_df is None:
                team_df = st.session_state['team_df']

            with st.spinner("🔄 Analyzing skills..."):
                # Compute analysis
                jd_skills = extract_skills_from_text(jd_text)
                team_skill_counts = aggregate_team_skills(team_df)
                score, matched, missing_detail = compute_skill_match(jd_skills, team_skill_counts)

                st.markdown("### 📊 Results Dashboard")
                
                # Metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric(label="Match Score", value=f"{score}%", delta=f"{score-50}% vs baseline")
                with col_m2:
                    st.metric(label="Matched Skills", value=len(matched))
                with col_m3:
                    st.metric(label="Missing Skills", value=len(missing_detail))

                # Details
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.markdown("**✅ Matched Skills**")
                    if matched:
                        for skill in matched:
                            st.markdown(f"- {skill}")
                    else:
                        st.write("None")
                
                with col_d2:
                    st.markdown("**⚠️ Missing / Low Coverage Skills**")
                    if missing_detail:
                        st.table(pd.DataFrame(missing_detail))
                    else:
                        st.write("None")

                # Radar chart
                st.markdown("### 📈 Skills Radar Visualization")
                viz_skills = list(jd_skills)[:8]
                team_size = max(1, sum(team_skill_counts.values()))
                viz_values = [int(100 * team_skill_counts.get(s,0) / team_size) for s in viz_skills]
                fig = radar_chart(viz_skills, viz_values)
                if fig:
                    st.pyplot(fig)

                # AI Summary
                st.markdown("### 🤖 AI-Powered Insights")
                prompt = f"""
Job Description skills: {', '.join(jd_skills)}
Matched skills: {', '.join(matched)}
Missing skills: {', '.join([m['skill'] for m in missing_detail])}

Provide a concise summary in bullet/point form (~200 words) focusing on:
- overall readiness
- major skill gaps
- recommended training/hiring actions
- 6-month outlook
"""
                with st.spinner("🤖 Generating AI summary..."):
                    summary = call_gemini(prompt, system_prompt="You are an expert HR AI. Output in bullet points, ~200 words.")
                st.markdown(summary)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("""
<div class="app-footer">
    <div class="footer-content">
        <div class="footer-section">
            <h3>About AI Skills Radar</h3>
            <p>Your intelligent workforce analytics platform that helps HR and L&D teams align job requirements with current team capabilities and plan upskilling strategies.</p>
        </div>
        <div class="footer-section">
            <h3>Contact Us</h3>
            <p>📧 Email: <a href="mailto:info@aiskillsradar.com">info@aiskillsradar.com</a></p>
            <p>📞 Phone: +1 (555) 123-4567</p>
            <p>💬 Support: <a href="mailto:support@aiskillsradar.com">support@aiskillsradar.com</a></p>
        </div>
        <div class="footer-section">
            <h3>Address</h3>
            <p>123 Tech Boulevard</p>
            <p>Innovation District</p>
            <p>San Francisco, CA 94105</p>
            <p>United States</p>
        </div>
    </div>
    <div class="footer-bottom">
        © 2025 AI Skills Radar | Empowering HR with AI-Driven Insights | All Rights Reserved
    </div>
</div>
""", unsafe_allow_html=True)
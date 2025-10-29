# streamlit_app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import requests
from io import StringIO
from PyPDF2 import PdfReader

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="AI Skills Radar", layout="wide", initial_sidebar_state="collapsed")

# ----------------------------
# Load Gemini key (secrets or env)
# ----------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
MODEL_NAME = "gemini-2.0-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# If missing, continue but show notice when trying to call the API
if not GEMINI_API_KEY:
    st.sidebar.warning("Gemini API key not found. Set GEMINI_API_KEY in Streamlit Secrets or environment variables.")

# ----------------------------
# CSS + JS (navbar fixed, footer fixed, styles)
# ----------------------------
st.markdown(
    """
    <style>
    /* Page background & font */
    html, body, [class*="css"]  {
        background: linear-gradient(180deg, #f8f5ff, #ffffff);
        font-family: 'Inter', 'Poppins', sans-serif;
    }

    /* Fixed navbar */
    .top-nav {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 28px;
        background: linear-gradient(90deg,#7c4dff 0%, #5ab9ff 100%);
        border-bottom-left-radius: 10px;
        border-bottom-right-radius: 10px;
        box-shadow: 0 6px 24px rgba(92,50,168,0.12);
    }
    .top-nav h1 {
        color: white;
        margin: 0;
        font-size: 18px;
        font-weight: 700;
    }
    .nav-buttons { display:flex; gap:10px; align-items:center; }
    .nav-btn {
        background: transparent;
        color: rgba(255,255,255,0.95);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 8px 12px;
        border-radius: 10px;
        cursor: pointer;
        font-weight:600;
        transition: transform .12s ease, background .12s ease;
    }
    .nav-btn:hover { transform: translateY(-2px); background: rgba(255,255,255,0.06); }
    .nav-btn.active { background: rgba(0,0,0,0.12); }

    /* main content to avoid navbar overlap */
    .main-block { padding: 96px 40px 96px 40px; }

    /* card */
    .card {
        background: white;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(100,80,160,0.06);
        margin-bottom: 18px;
    }

    /* footer */
    .app-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 12px 20px;
        text-align: center;
        font-size: 13px;
        color: #555;
        background: #fafafa;
        border-top: 1px solid #eee;
    }

    /* small responsive tweaks */
    @media (max-width: 700px) {
        .top-nav { padding: 10px 12px; }
        .main-block { padding: 120px 12px 120px 12px; }
        .nav-btn { padding: 6px 8px; font-size: 13px; }
    }
    </style>

    <script>
    // Small script to send a click event to Streamlit via the element's id (we use buttons rendered by Streamlit)
    // This script only adds a little hover animation if needed.
    document.addEventListener('DOMContentLoaded', function() {
      // nothing required here for core functionality; Streamlit buttons handle state server-side
    });
    </script>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Navigation (single page, SPA-like via session_state)
# ----------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "home"

# Render navbar using Streamlit elements inside a container so buttons set session_state
nav_container = st.container()
with nav_container:
    st.markdown('<div class="top-nav">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown('<h1>🤖 AI Skills Radar</h1>', unsafe_allow_html=True)
    with col2:
        # Create nav buttons horizontally
        nav_cols = st.columns([1,1,1,1,1,1])
        labels = ["Home", "Upload", "Dashboard", "Account", "Training", "About"]
        keys = ["nav_home", "nav_upload", "nav_dashboard", "nav_account", "nav_training", "nav_about"]
        for c, label, key in zip(nav_cols, labels, keys):
            if c.button(label, key=key):
                st.session_state["page"] = label.lower()
    st.markdown('</div>', unsafe_allow_html=True)

# spacer so top content not hidden
st.markdown('<div class="main-block">', unsafe_allow_html=True)

# ----------------------------
# Helper utils (skills, parsing, charts, LLM)
# ----------------------------
COMMON_SKILLS = [
    'python','sql','excel','data analysis','communication','project management',
    'leadership','product management','machine learning','nlp','aws','gcp','react',
    'java','c#','sales','negotiation','recruiting','interviewing','coaching','training'
]

def extract_skills_from_text(text, skill_list=COMMON_SKILLS):
    text_lower = (text or "").lower()
    found = set()
    for s in skill_list:
        if re.search(r'\b' + re.escape(s) + r'\b', text_lower):
            found.add(s)
    if not found:
        tokens = re.findall(r"[a-zA-Z]{4,}", text_lower)
        found = set(tokens[:8])
    return sorted(found)

def aggregate_team_skills(df_profiles):
    counts = {}
    if df_profiles is None or df_profiles.empty:
        return counts
    # if skills column exists (comma/semi/;) handle it
    if "skills" in [c.lower() for c in df_profiles.columns]:
        col = next(c for c in df_profiles.columns if c.lower() == "skills")
        for row in df_profiles[col].dropna():
            for s in [x.strip().lower() for x in re.split('[,;|/\\\\n]', str(row)) if x.strip()]:
                counts[s] = counts.get(s, 0) + 1
    else:
        for _, r in df_profiles.iterrows():
            combined = " ".join([str(x) for x in r.values if pd.notna(x)])
            for s in extract_skills_from_text(combined):
                counts[s] = counts.get(s, 0) + 1
    return counts

def compute_skill_match(jd_skills, team_skill_counts):
    jd_set = set(jd_skills)
    team_set = set(team_skill_counts.keys())
    matched = sorted(list(jd_set & team_set))
    missing = sorted(list(jd_set - team_set))
    score = int(100 * len(matched) / max(1, len(jd_set)))
    missing_detail = [{'skill': s, 'team_count': team_skill_counts.get(s,0)} for s in missing]
    return score, matched, missing_detail

def radar_chart(skills, values, title='Team Skills Radar'):
    if not skills:
        st.info("No skills to visualize.")
        return None
    N = len(skills)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    vals = list(values)
    vals += vals[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, vals, linewidth=2, color="#6f42c1")
    ax.fill(angles, vals, alpha=0.25, color="#6f42c1")
    ax.set_thetagrids(np.degrees(angles[:-1]), skills)
    ax.set_ylim(0,100)
    ax.set_title(title)
    return fig

def call_gemini_api(prompt, max_output_tokens=512, temperature=0.2):
    if not GEMINI_API_KEY:
        return "Gemini API key not configured. Set it in Streamlit Secrets or environment variables."

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens}
    }
    try:
        resp = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
        data = resp.json()
        if "candidates" in data:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        elif "error" in data:
            return f"API error: {data['error']}"
        else:
            return str(data)
    except Exception as e:
        return f"Request failed: {e}"

# ----------------------------
# Pages (single-page navigation)
# ----------------------------
page = st.session_state.get("page", "home")

if page == "home":
    st.header("AI Skills Radar")
    st.write("Analyze your job descriptions against your team's skills and get AI-powered recommendations.")
    st.markdown(
        """
        <div class="card">
            <h3 style="margin-top:0">How it works</h3>
            <ol>
                <li>Upload a Job Description (PDF or text) and a CSV of team profiles (with a 'skills' column).</li>
                <li>Go to Dashboard to view matching score, missing skills, and radar visualization.</li>
                <li>Use the Training page to get suggested trainings or request AI-generated mentoring plans.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "upload":
    st.header("Upload: Job Description & Team Profiles")
    colA, colB = st.columns(2)
    with colA:
        jd_file = st.file_uploader("Upload JD (PDF or TXT)", type=["pdf","txt"], key="jd_uploader")
        jd_text = st.text_area("Or paste JD text here", height=180, key="jd_textarea")
    with colB:
        team_file = st.file_uploader("Upload Team CSV / Excel (must include a 'skills' column)", type=["csv","xlsx","xls"], key="team_uploader")
        st.markdown("**Sample CSV format**: name,role,skills")
        if st.button("Load sample data"):
            csv = "name,role,skills\nAlice,Senior Data Engineer,Python;SQL;AWS\nBob,Data Analyst,SQL;Excel\nCarol,ML Engineer,Python;Machine Learning;NLP\n"
            st.session_state["team_df"] = pd.read_csv(StringIO(csv))
            st.session_state["jd_text"] = "Senior Data Engineer with Python, SQL, AWS experience."
            st.success("Sample data loaded into session.")

    if st.button("Process & Save Inputs"):
        # JD extraction
        final_jd = ""
        if jd_file is not None:
            if jd_file.type == "application/pdf":
                reader = PdfReader(jd_file)
                pages_text = [p.extract_text() for p in reader.pages]
                final_jd = "\n".join([t for t in pages_text if t])
            else:
                try:
                    final_jd = jd_file.getvalue().decode("utf-8")
                except Exception:
                    final_jd = st.session_state.get("jd_text", "")
        else:
            final_jd = st.session_state.get("jd_text", "")

        # team file
        if team_file is not None:
            try:
                if team_file.name.endswith(".csv"):
                    team_df = pd.read_csv(team_file)
                else:
                    team_df = pd.read_excel(team_file)
                st.session_state["team_df"] = team_df
            except Exception as e:
                st.error(f"Failed to read team file: {e}")
                team_df = None

        st.session_state["jd_text"] = final_jd
        st.success("Inputs processed and stored in session. Go to Dashboard.")

elif page == "dashboard":
    st.header("Dashboard: Skill Match & Gaps")
    jd_text = st.session_state.get("jd_text", "")
    team_df = st.session_state.get("team_df", None)

    if not jd_text or team_df is None:
        st.warning("Please upload JD and team profiles in Upload first. You can also load sample data there.")
    else:
        jd_skills = extract_skills_from_text(jd_text)
        team_skill_counts = aggregate_team_skills(team_df)
        score, matched, missing_detail = compute_skill_match(jd_skills, team_skill_counts)

        c1, c2, c3 = st.columns(3)
        c1.metric("Match Score", f"{score}%")
        c2.metric("Matched", len(matched))
        c3.metric("Missing", len(missing_detail))

        st.subheader("Matched skills")
        st.write(", ".join(matched) if matched else "None")

        st.subheader("Missing / Low coverage skills")
        st.dataframe(pd.DataFrame(missing_detail) if missing_detail else pd.DataFrame(columns=["skill","team_count"]))

        st.subheader("Team Coverage Radar")
        viz_skills = list(jd_skills)[:8]
        team_size = max(1, sum(team_skill_counts.values()))
        viz_values = [int(100 * team_skill_counts.get(s,0) / team_size) for s in viz_skills]

        fig = radar_chart(viz_skills, viz_values)
        if fig:
            st.pyplot(fig)

        if st.button("Generate AI Action Plan"):
            prompt = f"Job description skills: {jd_skills}\nTeam skills counts: {team_skill_counts}\nIdentify main gaps and propose a 6-month upskilling plan with suggested courses/mentors."
            with st.spinner("Calling Gemini..."):
                ai_text = call_gemini_api(prompt)
            st.subheader("AI Action Plan")
            st.write(ai_text)

elif page == "account":
    st.header("Account")
    # Combined Login & Signup tabs inside one page
    tab1, tab2 = st.tabs(["Login", "Signup"])
    with tab1:
        st.subheader("Login")
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                # For demo, simple check; replace with real auth later
                if user == "admin" and pwd == "1234":
                    st.success("Login successful (demo).")
                    st.session_state["user"] = user
                else:
                    st.error("Invalid credentials (demo).")
    with tab2:
        st.subheader("Signup")
        with st.form("signup_form"):
            new_user = st.text_input("Choose username")
            new_email = st.text_input("Email")
            new_pwd = st.text_input("Password", type="password")
            new_pwd_confirm = st.text_input("Confirm password", type="password")
            created = st.form_submit_button("Create account")
            if created:
                if new_pwd != new_pwd_confirm:
                    st.error("Passwords do not match.")
                elif not new_user or not new_email:
                    st.error("Please provide username and email.")
                else:
                    # In production, save to DB or auth service; here we store in session for demo
                    st.success(f"Account '{new_user}' created (demo).")
                    st.session_state["user"] = new_user

elif page == "training":
    st.header("Training Suggestions")
    missing_detail = st.session_state.get("missing_detail", None)
    # Allow user to generate suggestions from dashboard results or type skills manually
    skills_input = st.text_input("Enter skills to get suggested trainings (comma-separated)", value="")
    if st.button("Get Suggestions"):
        skills_list = [s.strip() for s in skills_input.split(",") if s.strip()]
        if not skills_list:
            st.info("Please enter at least one skill.")
        else:
            # Simple rule-based suggestions
            df_suggestions = pd.DataFrame([{
                "Skill": s,
                "Recommended Course": f"Intro to {s.title()} (external)",
                "Internal Mentor": f"Senior {s.title()} Specialist"
            } for s in skills_list])
            st.dataframe(df_suggestions)

elif page == "about":
    st.header("About")
    st.write("""
        AI Skills Radar — workforce planning tool for HR & L&D.
        Built with Streamlit and integrated with Gemini for AI-driven insights.
    """)

# close main block
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Footer (always fixed)
# ----------------------------
st.markdown(
    """
    <div class="app-footer">
        © 2025 AI Skills Radar • Built with ❤️ • support@aiskillsradar.com
    </div>
    """,
    unsafe_allow_html=True,
)

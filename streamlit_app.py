# JD Team Skill Gap Analyzer - Streamlit Demo
# Single-file Streamlit app (demo) with CSS + JavaScript and Gemini/LLM support.
# Supports entering API key at runtime (no TOML required).

import streamlit as st
import pandas as pd
import re
import json
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import StringIO

st.set_page_config(page_title="AI Skills Radar for Workforce Planning", layout="wide")

# ----------------------
# CSS (light purple theme)
# ----------------------
st.markdown("""
<style>
:root{
  --accent:#8a63d2;
  --accent-2:#b89cf6;
  --bg:#fbf8ff;
  --card:#ffffff;
}
body { background: var(--bg); }
[data-testid="stAppViewContainer"]{ background: linear-gradient(180deg, #fbf8ff, #f6f1ff); }
header {background: transparent}
.stTitle { color: #4b2d6f; font-weight:700; }
.card { background: var(--card); border-radius:14px; padding:18px; box-shadow: 0 6px 18px rgba(120,90,200,0.08); }
.small-muted{ color:#6b5e7a; font-size:13px }
.skill-pill { display:inline-block; padding:6px 10px; border-radius:999px; margin:4px; background:linear-gradient(90deg,var(--accent),var(--accent-2)); color:white }
</style>
""", unsafe_allow_html=True)

# ----------------------
# JavaScript (copy-to-clipboard)
# ----------------------
st.components.v1.html("""
<script>
function copyText(id){
  const el = document.getElementById(id);
  if(!el) return;
  navigator.clipboard.writeText(el.innerText).then(()=>{
    alert('Copied to clipboard');
  })
}
</script>
""", height=0)

# ----------------------
# API Key input (runtime)
# ----------------------
GEMINI_API_KEY = st.text_input("Enter Gemini API Key", type="password")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
GEMINI_MODEL = "gemini-1.5-flash"

# ----------------------
# LLM wrapper
# ----------------------
def call_llm(prompt, max_tokens=512, temperature=0.0):
    if not GEMINI_API_KEY:
        return {"text": "No API key provided. Enter it in the box above."}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}"
    }

    payload = {
        "prompt": [{"content": prompt, "type": "text"}],
        "temperature": temperature,
        "candidate_count": 1,
        "max_output_tokens": max_tokens
    }

    try:
        resp = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]
        return {"text": text}
    except Exception as e:
        return {"text": f"ERROR_CALLING_LLM: {str(e)}"}

# ----------------------
# Skill parsing helpers
# ----------------------
COMMON_SKILLS = [
    'python','sql','excel','data analysis','communication','project management',
    'leadership','product management','machine learning','nlp','aws','gcp','react',
    'java','c#','sales','negotiation','recruiting','interviewing','coaching','training'
]

def extract_skills_from_text(text, skill_list=COMMON_SKILLS):
    text_lower = text.lower()
    found = set()
    for s in skill_list:
        if re.search(r'\b' + re.escape(s) + r'\b', text_lower):
            found.add(s)
    if not found:
        tokens = re.findall(r"[a-zA-Z]{4,}", text_lower)
        found = set(tokens[:8])
    return sorted(found)

def aggregate_team_skills(df_profiles, skill_list=COMMON_SKILLS):
    counts = {}
    if 'skills' in df_profiles.columns:
        for row in df_profiles['skills'].dropna():
            for s in [x.strip().lower() for x in re.split('[,;|/\\\\n]', str(row)) if x.strip()]:
                counts[s] = counts.get(s, 0) + 1
    else:
        for _, r in df_profiles.iterrows():
            combined = ' '.join([str(x) for x in r.values if pd.notna(x)])
            for s in extract_skills_from_text(combined, skill_list):
                counts[s] = counts.get(s, 0) + 1
    return counts

def compute_skill_match(jd_skills, team_skills_counts):
    jd_set = set(jd_skills)
    team_set = set(team_skills_counts.keys())
    matched = jd_set & team_set
    missing = jd_set - team_set
    score = int(100 * len(matched) / max(1, len(jd_set)))
    missing_detail = [{ 'skill': s, 'team_count': team_skills_counts.get(s,0) } for s in sorted(missing)]
    return score, sorted(matched), missing_detail

def suggest_training_for_skills(skills):
    suggestions = []
    for s in skills:
        suggestions.append({
            'skill': s,
            'recommended_course': f'Intro to {s.title()} (external course)',
            'internal_mentor': f'Mentor: Senior {s.title()} Specialist' if s not in ['communication','coaching'] else 'Mentor: L&D Coach'
        })
    return suggestions

TRENDING_SKILLS = ['machine learning','nlp','cloud','gcp','aws','data analysis','python','automation']

def predict_future_gaps(team_skill_counts, months=6):
    future_need = {}
    team_size = max(1, sum(team_skill_counts.values()))
    for s in TRENDING_SKILLS:
        coverage = team_skill_counts.get(s, 0) / team_size
        need = coverage < 0.3
        future_need[s] = { 'coverage': round(coverage,2), 'predicted_gap': need }
    return future_need

def radar_chart(skills, values, title='Team Skills Radar'):
    labels = skills
    stats = values
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    stats = list(stats)
    stats += stats[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, stats, linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0,100)
    ax.set_title(title)
    return fig

# ----------------------
# UI
# ----------------------
st.title('AI Skills Radar for Workforce Planning', anchor=None)
st.caption('JD Team Skill Gap Analyzer — demo (Phase 1)')

left, right = st.columns([1,1])
with left:
    st.markdown('<div class="card"> <h3 class="stTitle">Upload Inputs</h3>', unsafe_allow_html=True)
    jd_file = st.file_uploader('Upload Job Description (txt or pdf)', type=['txt','pdf'], key='jd_upload')
    jd_text = st.text_area('Or paste JD text here', height=160)
    team_file = st.file_uploader('Upload Team Profiles (CSV / Excel) — must contain name + skills columns', type=['csv','xlsx','xls'], key='team_upload')
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Tip: "skills" column can be comma separated (python, sql, leadership)</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card"> <h3 class="stTitle">Quick Options</h3>', unsafe_allow_html=True)
    st.checkbox('Enable predictive 6-month gap analysis', value=True, key='predict_toggle')
    st.selectbox('Confidence level for LLM-assisted extraction', ['Low','Medium','High'], index=1)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('---')

# Load JD text
if jd_file is not None:
    if jd_file.type == 'text/plain':
        jd_text = jd_file.getvalue().decode('utf-8')
    else:
        jd_text = '(Uploaded PDF — text extraction not implemented in demo)\n' + (jd_text or '')

# Load team profiles
team_df = None
if team_file is not None:
    try:
        if team_file.name.endswith('.csv'):
            team_df = pd.read_csv(team_file)
        else:
            team_df = pd.read_excel(team_file)
    except Exception as e:
        st.error('Failed to read team file: ' + str(e))

# Demo data if no inputs
if not jd_text and team_df is None:
    st.info('No inputs detected — using demo JD and team profiles.')
    jd_text = """
    We are hiring a Senior Data Engineer with strong Python, SQL, AWS, data pipeline design, and communication skills.
    Familiarity with Spark, Airflow, and performance tuning is a plus.
    """
    demo_csv = """name,role,skills
Alice,Senior Data Engineer,Python;SQL;AWS;Airflow
Bob,Data Analyst,SQL;Excel;Data Analysis
Carol,ML Engineer,Python;Machine Learning;NLP
Dan,Intern,Excel;Communication
"""
    team_df = pd.read_csv(StringIO(demo_csv))

# Extract skills
with st.spinner('Analyzing...'):
    jd_skills = extract_skills_from_text(jd_text)
    team_skill_counts = aggregate_team_skills(team_df)
    score, matched, missing_detail = compute_skill_match(jd_skills, team_skill_counts)

# Dashboard
col1, col2 = st.columns([1,1])
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('JD Skills Extracted')
    st.write(', '.join(jd_skills) if jd_skills else '—')
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('Team Snapshot')
    st.dataframe(pd.DataFrame([{'skill':k,'count':v} for k,v in sorted(team_skill_counts.items(), key=lambda x:-x[1])]))
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('---')

# Skill match summary
st.subheader('Match Summary')
st.metric(label='JD → Team Match Score', value=f'{score}%', delta=f'{len(matched)}/{len(jd_skills)} skills matched')
st.write('**Matched skills:**', ', '.join(matched) if matched else 'None')
if missing_detail:
    st.write('**Missing / Low coverage skills:**')
    st.table(pd.DataFrame(missing_detail))

# Radar visualization
viz_skills = list(jd_skills)[:8] if jd_skills else list(team_skill_counts.keys())[:8]
viz_values = []
team_size = max(1, sum(team_skill_counts.values()))
for s in viz_skills:
    viz_values.append(int(100 * team_skill_counts.get(s,0) / team_size))

if viz_skills:
    fig = radar_chart(viz_skills, viz_values, title='Team coverage vs JD')
    st.pyplot(fig)

# Training suggestions
if missing_detail:
    st.subheader('Recommended Trainings / Mentoring')
    suggestions = suggest_training_for_skills([m['skill'] for m in missing_detail])
    st.table(pd.DataFrame(suggestions))

# Predictive gap analysis
if st.session_state.get('predict_toggle'):
    st.subheader('Predictive 6-month Gap Analysis')
    future = predict_future_gaps(team_skill_counts, months=6)
    df_future = pd.DataFrame([{ 'skill':k, 'coverage':v['coverage'], 'predicted_gap':v['predicted_gap']} for k,v in future.items()])
    st.dataframe(df_future)

# LLM-assisted explanation
st.markdown('---')
if st.button('Get LLM-assisted summary & action plan'):
    prompt = f"Given the JD:\n{jd_text}\n\nTeam skills counts:\n{json.dumps(team_skill_counts)}\n\nProvide a short action plan describing main gaps, recommended trainings, and a 6-month upskilling roadmap."
    with st.spinner('Calling LLM...'):
        resp = call_llm(prompt)
        st.subheader('LLM Action Plan')
        st.write(resp.get('text','(no response)'))

# Export summary
st.markdown('---')
if st.button('Export summary to clipboard (demo)'):
    st.components.v1.html('<div id="summaryText" style="display:none">'+st.session_state.get('last_summary','Demo Summary')+'</div>', height=0)
    st.components.v1.html('<script>copyText("summaryText")</script>', height=0)
    st.success('Summary copy triggered (browser permission required).')

st.markdown('\n\n---\n\n**Notes:** This is a demo app. PDF text extraction and production-grade parsing/validation are omitted for brevity.')

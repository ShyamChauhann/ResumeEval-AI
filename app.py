import streamlit as st
import os
import re
import plotly.graph_objects as go
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# --- Config & Initialization ---
hf_token = os.getenv("HF_TOKEN")
UPLOAD_DIR = "uploaded_resumes"

st.set_page_config(
    page_title="ResumeEval AI", 
    page_icon="📄", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Badges, Metric Styling) ---
st.markdown("""
<style>
    [data-testid="stAppDeployButton"] { display: none; }
    
    /* Custom Skill Badges */
    .badge-success {
        background-color: rgba(38, 166, 154, 0.2);
        color: #26a69a;
        border: 1px solid #26a69a;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin: 3px;
    }
    .badge-warning {
        background-color: rgba(255, 152, 0, 0.2);
        color: #ffa726;
        border: 1px solid #ffa726;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin: 3px;
    }
    
    /* Highlighted card boxes */
    .report-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.title("⚙️ Settings")

if not hf_token:
    st.sidebar.warning("`HF_TOKEN` secret not found in environment.")
    hf_token = st.sidebar.text_input("Enter Hugging Face Token (hf_...):", type="password")

if hf_token:
    client = InferenceClient(model="Qwen/Qwen2.5-72B-Instruct", token=hf_token)
else:
    client = None

target_role = st.sidebar.selectbox(
    "Select Target Field:",
    [
        "Machine Learning Engineer",
        "Data Scientist",
        "Data Analyst",
        "AI Research Engineer",
        "NLP Engineer",
        "Computer Vision Engineer",
        "MLOps Engineer"
    ]
)

# --- Helper Functions ---
def save_uploaded_file(uploaded_file):
    """Saves the uploaded file locally."""
    try:
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def extract_text_from_pdf(uploaded_file):
    """Extracts text from uploaded PDF."""
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def parse_scores(report_text):
    """Extracts numeric scores accurately from bulleted output."""
    ats_score = 0
    role_fit = 0.0
    
    # Matches: "- **ATS Score:** 75%" or "ATS Score: 75%"
    ats_match = re.search(r"ATS\s*Score:?\*?\*?\s*(\d{1,3})\s*%", report_text, re.IGNORECASE)
    if ats_match:
        ats_score = int(ats_match.group(1))
    else:
        # Backup search if % sign is missing
        backup_ats = re.search(r"ATS\s*Score:?\*?\*?\s*(\d{1,3})", report_text, re.IGNORECASE)
        if backup_ats:
            ats_score = int(backup_ats.group(1))

    # Matches: "- **Role Fit Score:** 7/10" or "Role Fit Score: 7.5"
    fit_match = re.search(r"Role\s*Fit\s*Score:?\*?\*?\s*(\d+(?:\.\d+)?)\s*(?:/\s*10)?", report_text, re.IGNORECASE)
    if fit_match:
        role_fit = float(fit_match.group(1))
            
    return ats_score, role_fit

def render_gauge_chart(score, title, max_val=100, suffix="%"):
    """Renders a correctly scaled Plotly gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': suffix, 'font': {'size': 32}},
        title={'text': title, 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1, 'dtick': max_val / 2},
            'bar': {'color': "#26a69a" if score >= (max_val * 0.7) else "#ffa726" if score >= (max_val * 0.5) else "#ef5350"},
            'bgcolor': "rgba(0,0,0,0.1)",
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_val * 0.5], 'color': 'rgba(239, 83, 80, 0.15)'},
                {'range': [max_val * 0.5, max_val * 0.7], 'color': 'rgba(255, 167, 38, 0.15)'},
                {'range': [max_val * 0.7, max_val], 'color': 'rgba(38, 166, 154, 0.15)'}
            ],
        }
    ))
    fig.update_layout(height=220, margin=dict(l=25, r=25, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def render_badges(items, badge_type="success"):
    """Turns comma-separated skill lists into styled inline badges."""
    if not items:
        return ""
    skills = [s.strip() for s in items.split(",") if s.strip()]
    badge_html = "".join([f'<span class="badge-{badge_type}">{skill}</span>' for skill in skills])
    return badge_html

def analyze_resume(resume_text, role):
    if not client:
        return "Error: Hugging Face Token is missing."

    prompt = f"""
    You are an expert ATS scanner and a senior technical recruiter specializing in AI and Data Science.
    
    Analyze this resume for the position of: "{role}".
    
    CRITICAL INSTRUCTION: You MUST follow the exact template structure below including exact headers.
    
    ### Summary Scores
    - **ATS Score:** [Provide percentage score e.g., 78%]
    - **Role Fit Score:** [Provide score out of 10 e.g., 8/10]

    ### Skills Extraction
    - **Found Critical Skills:** [List matching technical skills separated strictly by commas, e.g. Python, PyTorch, SQL]
    - **Missing/Weak Skills:** [List missing skills separated strictly by commas, e.g. MLOps, Docker, Kubernetes]

    ### Project & Experience Review
    Provide structured feedback on project depth, metrics, and deployment experience. Use bold headers where appropriate.

    ### Actionable Improvement Checklist
    Provide 3 to 5 clear, bullet points on specific improvements.

    Resume Text:
    {resume_text}
    """

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

# --- Main App Interface ---
st.title("📄 AI Resume Analyzer & ATS Evaluator")
st.markdown("Upload your resume to get an **interactive analysis dashboard** with ATS scoring, skill gap detection, and actionable feedback.")

uploaded_file = st.file_uploader("Upload your Resume (PDF format only)", type=["pdf"])

if uploaded_file is not None:
    saved_path = save_uploaded_file(uploaded_file)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"📁 **File Ready:** `{uploaded_file.name}`")
    with col2:
        analyze_btn = st.button("🚀 Analyze Resume", type="primary", use_container_width=True)

    if analyze_btn:
        if not hf_token:
            st.error("Please enter a valid Hugging Face Token in the sidebar.")
        else:
            with st.spinner("⚡ Extracting text and parsing with Qwen 2.5 72B..."):
                resume_text = extract_text_from_pdf(uploaded_file)
                
                if not resume_text.strip():
                    st.error("Could not extract readable text. Ensure it is not a scanned/image-only PDF.")
                else:
                    report_text = analyze_resume(resume_text, target_role)
                    
                    st.markdown("---")
                    st.header(f"📊 Dashboard Analysis for {target_role}")
                    
                    # --- Dashboard Metrics Area ---
                    ats_score, role_fit = parse_scores(report_text)
                    
                    gauge_col1, gauge_col2, metric_col3 = st.columns([1, 1, 1])

                    with gauge_col1:
                        st.plotly_chart(render_gauge_chart(ats_score, "ATS Match Score", max_val=100, suffix="%"), use_container_width=True)

                    with gauge_col2:
                        st.plotly_chart(render_gauge_chart(role_fit, "Role Suitability", max_val=10, suffix="/10"), use_container_width=True)
                        
                    with metric_col3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.metric("Target Role", target_role)
                        if ats_score >= 70:
                            st.success("🎯 High Match Probability")
                        elif ats_score >= 50:
                            st.warning("⚠️ Moderate Match - Needs Tweaks")
                        else:
                            st.error("🚨 Low Match - Significant Gaps")

                    # --- Tabbed Sections for Structured Feedback ---
                    tab_skills, tab_review, tab_action, tab_raw = st.tabs([
                        "🛠️ Skills & Gap Analysis", 
                        "🔍 Experience & Project Review", 
                        "✅ Action Plan", 
                        "📝 Full Text Report"
                    ])

                    with tab_skills:
                        st.subheader("Skill Match Dashboard")
                        
                        # Extract skill strings via regex for badging
                        found_match = re.search(r"Found Critical Skills:\*\*?\s*(.*?)(?=\n|- \*\*Missing|\Z)", report_text, re.IGNORECASE | re.DOTALL)
                        missing_match = re.search(r"Missing/Weak Skills:\*\*?\s*(.*?)(?=\n|###|\Z)", report_text, re.IGNORECASE | re.DOTALL)
                        
                        found_str = found_match.group(1).strip() if found_match else ""
                        missing_str = missing_match.group(1).strip() if missing_match else ""

                        col_f, col_m = st.columns(2)
                        with col_f:
                            st.markdown("### ✅ Identified Key Skills")
                            if found_str:
                                st.markdown(render_badges(found_str, "success"), unsafe_allow_html=True)
                            else:
                                st.write("No matching key skills parsed.")
                        
                        with col_m:
                            st.markdown("### ⚠️ Missing / Weak Skills")
                            if missing_str:
                                st.markdown(render_badges(missing_str, "warning"), unsafe_allow_html=True)
                            else:
                                st.write("No major skill gaps detected!")

                    with tab_review:
                        st.subheader("Project & Technical Depth Review")
                        review_match = re.search(r"### Project & Experience Review\n*(.*?)(?=### Actionable Improvement Checklist|\Z)", report_text, re.DOTALL)
                        review_text = review_match.group(1) if review_match else "Review details included in full report."
                        st.info(review_text)

                    with tab_action:
                        st.subheader("Actionable Checklist")
                        action_match = re.search(r"### Actionable Improvement Checklist\n*(.*?)\Z", report_text, re.DOTALL)
                        action_text = action_match.group(1) if action_match else "Checklist included in full report."
                        st.write(action_text)

                    with tab_raw:
                        st.subheader("Raw AI Model Output")
                        st.markdown(report_text)
# import streamlit as st
# import os
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv
# from huggingface_hub import InferenceClient
# load_dotenv()
# # 1. Fetch the Hugging Face Token automatically from Space Secrets
# # By default, HF Spaces often provides 'HF_TOKEN' if enabled in settings
# hf_token = os.getenv("HF_TOKEN")
# UPLOAD_DIR = "uploaded_resumes"

# # --- Streamlit UI Setup ---
# st.set_page_config(page_title="ResumeEval AI", page_icon="📄", layout="wide")

# st.markdown("""
# <style>
# [data-testid="stAppDeployButton"] {
#     display: none;
# }
# </style>
# """, unsafe_allow_html=True)

# st.title("Resume Analyzer")
# st.markdown("Upload your resume to get an instant ATS score, role suitability rating, and optimization tips.")

# # 2. Sidebar fallback input if the environment variable isn't found
# if not hf_token:
#     st.sidebar.warning("`HF_TOKEN` secret not found in HF Space.")
#     hf_token = st.sidebar.text_input("Enter Hugging Face Token (hf_...):", type="password")

# # Configure the Hugging Face Client
# if hf_token:
#     # Using Qwen 2.5 72B Instruct for high-quality structured analysis
#     client = InferenceClient(model="Qwen/Qwen2.5-72B-Instruct", token=hf_token)
# else:
#     client = None

# target_role = st.sidebar.selectbox(
#     "Select your Preferred Field:",
#     [
#         "Machine Learning Engineer",
#         "Data Scientist",
#         "Data Analyst",
#         "AI Research Engineer",
#         "NLP Engineer",
#         "Computer Vision Engineer",
#         "MLOps Engineer"

#     ]
# )


# def save_uploaded_file(uploaded_file):
#     """Saves the uploaded file to the local directory."""
#     try:
#         # Create directory if it doesn't exist
#         if not os.path.exists(UPLOAD_DIR):
#             os.makedirs(UPLOAD_DIR)
        
#         # Define the full file path
#         file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        
#         # Write the file bytes locally
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
            
#         return file_path
#     except Exception as e:
#         st.error(f"Error saving file: {e}")
#         return None

# # --- Main Streamlit Upload Logic ---
# uploaded_file = st.file_uploader("Upload your Resume (PDF format only)", type=["pdf"])

# if uploaded_file is not None:
#     # Save the file locally when uploaded
#     saved_path = save_uploaded_file(uploaded_file)
    
#     if saved_path:
#         st.success(f"File saved locally at: `{saved_path}`")

# def extract_text_from_pdf(uploaded_file):
#     """Extracts all text content from an uploaded PDF file."""
#     pdf_reader = PdfReader(uploaded_file)
#     text = ""
#     for page in pdf_reader.pages:
#         page_text = page.extract_text()
#         if page_text:
#             text += page_text + "\n"
#     return text

# def analyze_resume(resume_text, role):
#     # Sends the resume text to Hugging Face API for an ATS analysis
#     if not client:
#         return "Error: Hugging Face Token is missing. Please set it via Space Secrets or the sidebar."

#     prompt = f"""
#     You are an expert ATS (Applicant Tracking System) scanner and a technical recruiter specializing in AI, Machine Learning, and Data Science roles.
    
#     Analyze the following resume text specifically for suitability towards the preferred field/role: "{role}".
    
#     Evaluate the resume based on:
#     1. Keywords match (e.g., Python, SQL, PyTorch, TensorFlow, Scikit-Learn, MLOps, NLP, Computer Vision, LLMs, cloud platforms).
#     2. Project depth (e.g., data pipelines, model deployment, evaluation metrics used).
#     3. Formatting, readability, and impact metrics.

#     Provide your response in a clean, highly scannable Markdown format with the following explicit sections:
    
#     ### Summary Scores
#     - **ATS Score:** [Provide an objective percentage score out of 100%]
#     - **Role Fit Score:** [Provide a recommended rating out of 10 for the role "{role}"]

#     ### Keyword & Skills Gap Analysis
#     - **Found Critical Skills:** [List key technical skills matching the role found in the resume]
#     - **Missing/Weak Skills:** [Identify crucial AI/ML/Data Science concepts or tools missing that would strengthen the application]

#     ### Project & Experience Review
#     - [Provide constructive feedback on technical depth, metrics, and whether they demonstrated end-to-end implementation/deployment]

#     ### Actionable Improvement Checklist
#     - [Provide 3-4 bullet points on exactly what to change, add, or rewrite to maximize impact]
    
#     Resume Content:
#     {resume_text}
#     """

#     try:
#         # Generate completion using Hugging Face's Chat Completion API
#         messages = [{"role": "user", "content": prompt}]
#         response = client.chat.completions.create(
#             messages=messages,
#             max_tokens=2000,
#             stream=False
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"An error occurred while communicating with the Hugging Face model: {str(e)}"

# # # Main layout upload area
# # uploaded_file = st.file_uploader("Upload your Resume (PDF format only)", type=["pdf"])

# if uploaded_file is not None:
#     st.success("Resume uploaded successfully!")
    
#     # Analyze button
#     if st.button("Analyze Resume", type='primary'):
#         if not hf_token:
#             st.error("Cannot analyze.Please Enter a valid Hugging Face Token first.")
#         else:
#             with st.spinner("Analyzing resume against industry standards... Please wait."):
#                 # 1. Extract text from the PDF
#                 resume_text = extract_text_from_pdf(uploaded_file)
                
#                 if not resume_text.strip():
#                     st.error("Could not extract text from the PDF. Please ensure it is not an image-only/scanned PDF.")

#                 else:
#                     # 2. Query Hugging Face API
#                     analysis_report = analyze_resume(resume_text, target_role)
                    
#                     # 3. Display Results
#                     st.markdown("---")
#                     st.subheader(f"Analysis Report for {target_role}")
#                     st.markdown(analysis_report)
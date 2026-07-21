import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
load_dotenv()
# 1. Fetch the Hugging Face Token automatically from Space Secrets
# By default, HF Spaces often provides 'HF_TOKEN' if enabled in settings
hf_token = os.getenv("HF_TOKEN")

# --- Streamlit UI Setup ---
st.set_page_config(page_title="ResumeEval AI", page_icon="📄", layout="wide")

st.title("Resume Analyzer")
st.markdown("Upload your resume to get an instant ATS score, role suitability rating, and optimization tips.")

# 2. Sidebar fallback input if the environment variable isn't found
if not hf_token:
    st.sidebar.warning("`HF_TOKEN` secret not found in HF Space.")
    hf_token = st.sidebar.text_input("Enter Hugging Face Token (hf_...):", type="password")

# Configure the Hugging Face Client
if hf_token:
    # Using Qwen 2.5 72B Instruct for high-quality structured analysis
    client = InferenceClient(model="Qwen/Qwen2.5-72B-Instruct", token=hf_token)
else:
    client = None

target_role = st.sidebar.selectbox(
    "Select your Preferred Field:",
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

def extract_text_from_pdf(uploaded_file):
    """Extracts all text content from an uploaded PDF file."""
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def analyze_resume(resume_text, role):
    # Sends the resume text to Hugging Face API for an ATS analysis
    if not client:
        return "Error: Hugging Face Token is missing. Please set it via Space Secrets or the sidebar."

    prompt = f"""
    You are an expert ATS (Applicant Tracking System) scanner and a technical recruiter specializing in AI, Machine Learning, and Data Science roles.
    
    Analyze the following resume text specifically for suitability towards the preferred field/role: "{role}".
    
    Evaluate the resume based on:
    1. Keywords match (e.g., Python, SQL, PyTorch, TensorFlow, Scikit-Learn, MLOps, NLP, Computer Vision, LLMs, cloud platforms).
    2. Project depth (e.g., data pipelines, model deployment, evaluation metrics used).
    3. Formatting, readability, and impact metrics.

    Provide your response in a clean, highly scannable Markdown format with the following explicit sections:
    
    ### Summary Scores
    - **ATS Score:** [Provide an objective percentage score out of 100%]
    - **Role Fit Score:** [Provide a recommended rating out of 10 for the role "{role}"]

    ### Keyword & Skills Gap Analysis
    - **Found Critical Skills:** [List key technical skills matching the role found in the resume]
    - **Missing/Weak Skills:** [Identify crucial AI/ML/Data Science concepts or tools missing that would strengthen the application]

    ### Project & Experience Review
    - [Provide constructive feedback on technical depth, metrics, and whether they demonstrated end-to-end implementation/deployment]

    ### Actionable Improvement Checklist
    - [Provide 3-4 bullet points on exactly what to change, add, or rewrite to maximize impact]
    
    Resume Content:
    {resume_text}
    """

    try:
        # Generate completion using Hugging Face's Chat Completion API
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            messages=messages,
            max_tokens=2000,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while communicating with the Hugging Face model: {str(e)}"

# Main layout upload area
uploaded_file = st.file_uploader("Upload your Resume (PDF format only)", type=["pdf"])

if uploaded_file is not None:
    st.success("Resume uploaded successfully!")
    
    # Analyze button
    if st.button("Analyze Resume", type='primary'):
        if not hf_token:
            st.error("Cannot analyze.Please Enter a valid Hugging Face Token first.")
        else:
            with st.spinner("Analyzing resume against industry standards... Please wait."):
                # 1. Extract text from the PDF
                resume_text = extract_text_from_pdf(uploaded_file)
                
                if not resume_text.strip():
                    st.error("Could not extract text from the PDF. Please ensure it is not an image-only/scanned PDF.")

                else:
                    # 2. Query Hugging Face API
                    analysis_report = analyze_resume(resume_text, target_role)
                    
                    # 3. Display Results
                    st.markdown("---")
                    st.subheader(f"Analysis Report for {target_role}")
                    st.markdown(analysis_report)
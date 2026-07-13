# 📄 ResumeEval AI

> An AI-powered Resume Analyzer that evaluates resumes for AI/ML and Data Science roles using Large Language Models (LLMs), providing ATS scores, role-fit analysis, skill-gap identification, and actionable improvement recommendations.

---

## 🚀 Overview

ResumeEval AI is a Streamlit-based web application that helps candidates optimize their resumes for technical roles such as:

* Machine Learning Engineer
* Data Scientist
* Data Analyst
* AI Research Engineer
* NLP Engineer
* Computer Vision Engineer
* MLOps Engineer

The application extracts text from uploaded resumes, leverages the **Qwen 2.5 72B Instruct** Large Language Model through the Hugging Face Inference API, and generates structured resume evaluations with ATS-oriented feedback.

---

## ✨ Features

* 📄 Upload PDF resumes
* 🤖 AI-powered resume analysis using LLMs
* 📊 ATS Score (0–100)
* ⭐ Role Fit Score (0–10)
* 🔍 Keyword and Skill Gap Analysis
* 🛠 Technical Project Evaluation
* 💡 Personalized Resume Improvement Suggestions
* 🎯 Role-specific evaluation for AI/ML careers
* 🔐 Secure API key management using environment variables
* 🐳 Docker support for containerized deployment
* 💻 Interactive Streamlit interface

---

## 🛠 Tech Stack

### Programming Language

* Python

### Frontend

* Streamlit

### AI / LLM

* Qwen 2.5 72B Instruct
* Hugging Face Inference API

### PDF Processing

* PyPDF2

### Configuration

* python-dotenv

### Deployment

* Docker

### Version Control

* Git
* GitHub

---

## 📂 Project Structure

```text
ResumeEval-AI/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── .env
├── README.md
├── assets/
│
└── sample_resume.pdf
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/<your-username>/ResumeEval-AI.git

cd ResumeEval-AI
```

### Create a virtual environment

#### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

#### macOS/Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```text
HF_TOKEN="your_huggingface_token"
```

Generate your Hugging Face access token from your Hugging Face account settings.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will start at

```text
http://localhost:8501
```

---


---

## 📊 Workflow

```text
User Uploads Resume
          │
          ▼
PDF Text Extraction (PyPDF2)
          │
          ▼
Prompt Engineering
          │
          ▼
Qwen 2.5 72B Instruct (Hugging Face)
          │
          ▼
Resume Analysis
          │
          ▼
ATS Score
Role Fit Score
Skill Gap Analysis
Project Review
Improvement Suggestions
```

---

## 📸 Screenshots

Add screenshots of:

* Home Page
* Resume Upload
* Analysis Report
* ATS Score
* Role Fit Score

---

## 🎯 Supported Roles

* Machine Learning Engineer
* Data Scientist
* Data Analyst
* AI Research Engineer
* NLP Engineer
* Computer Vision Engineer
* MLOps Engineer

---

## 📈 Resume Highlights

* Developed an AI-powered resume analysis platform using Python, Streamlit, Docker, and Large Language Models.
* Built an end-to-end resume processing pipeline with PDF parsing, prompt engineering, and AI-generated ATS evaluation.
* Implemented secure API key management, modular architecture, and containerized deployment using Docker.

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push to your branch.
5. Open a Pull Request.

---

## 👨‍💻 Author

**Shyam Chauhan**

* GitHub: https://github.com/ShyamChauhann
* LinkedIn: https://www.linkedin.com/in/shyam-chauhan-221a9823a/

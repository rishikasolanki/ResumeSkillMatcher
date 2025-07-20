import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
from pdfminer.high_level import extract_text

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ----- Helper Functions -----
def extract_text_from_pdf(uploaded_file):
    if uploaded_file is not None:
        text = extract_text(uploaded_file)
        return text
    return ""

def extract_skills(text):
    doc = nlp(text.lower())
    skills = []
    keywords = ["python", "machine learning", "data analysis", "excel", "deep learning",
                "nlp", "sql", "communication", "leadership", "teamwork", "java", "c++", "time management"]
    for token in doc:
        if token.text in keywords:
            skills.append(token.text)
    return list(set(skills))

def calculate_similarity(resume_text, jd_text):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors[0:1], vectors[1:2])
    return round(float(score[0][0]) * 100, 2)

# ----- Streamlit App -----
st.set_page_config(page_title="Resume Skill Matcher", layout="centered")

st.title("🤖 Resume Skill Matcher")
st.markdown("Match your resume against a job description using AI.")

# File Upload
resume_file = st.file_uploader("Upload your Resume (PDF/Text)", type=["pdf", "txt"])
jd_file = st.file_uploader("Upload Job Description (PDF/Text)", type=["pdf", "txt"])

if st.button("Match Skills"):
    if resume_file and jd_file:
        # Extract text
        resume_text = extract_text_from_pdf(resume_file) if resume_file.name.endswith("pdf") else resume_file.read().decode("utf-8")
        jd_text = extract_text_from_pdf(jd_file) if jd_file.name.endswith("pdf") else jd_file.read().decode("utf-8")
        
        # Extract skills
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)
        
        # Match score
        match_score = calculate_similarity(" ".join(resume_skills), " ".join(jd_skills))

        # Missing skills
        missing_skills = list(set(jd_skills) - set(resume_skills))

        # Display
        st.subheader("🔍 Match Result")
        st.write(f"**Similarity Score:** {match_score} %")
        st.write("✅ **Your Skills:**", ", ".join(resume_skills))
        st.write("🎯 **Required Skills:**", ", ".join(jd_skills))
        st.write("❌ **Missing Skills:**", ", ".join(missing_skills) if missing_skills else "None! Great match 🎉")

    else:
        st.warning("Please upload both resume and job description files.")


from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import uvicorn
import numpy as np
import pandas as pd

# --------- Dataset ---------
data = {
    'name': ['Ishank', 'Lakshay', 'Priyanshi', 'Puneet', 'Aarti', 'Akansha', 'Sakshi', 'Peeyush', 'Shubham', 'Kannu'],
    'skills': [
        'Python, Machine Learning, Deep Learning',
        'HTML, CSS, JavaScript',
        'Python, NLP, Data Science',
        'C++, Java, SQL',
        'Python, Artificial Intelligence, FastAPI',
        'HTML, React, Node.js',
        'Python, Computer Vision, TensorFlow',
        'C, C++, Embedded Systems',
        'Python, ChatGPT, LLMs',
        'Python, GenAI, FastAPI',
    ],
    'is_ai_ml_ready': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

# --------- Load Model and Vectorizer ---------
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# --------- FastAPI App ---------
app = FastAPI(title="Resume AI/ML Readiness Predictor 🚀")

# Define Request Body Schema
class ResumeSkills(BaseModel):
    skills: str

# Root Endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Resume AI/ML Readiness Predictor 🚀! Check out /docs for the API."}

# Favicon Endpoint (optional)
@app.get("/favicon.ico")
def get_favicon():
    return FileResponse("favicon.ico")  # Remove if no favicon.ico

# Debug Endpoint to Check Vectorizer Output
@app.post("/debug_vectorizer")
def debug_vectorizer(data: ResumeSkills):
    X = vectorizer.transform([data.skills])
    return {
        "skills": data.skills,
        "vectorized_features": X.toarray().tolist(),
        "feature_names": vectorizer.get_feature_names_out().tolist()
    }

# Predict Endpoint
@app.post("/predict")
def predict_resume(data: ResumeSkills):
    # Transform input skills
    X = vectorizer.transform([data.skills])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X).max()

    # Find student name by matching skills
    student_name = "Unknown"
    matching_student = df[df['skills'].str.lower() == data.skills.lower()]
    if not matching_student.empty:
        student_name = matching_student['name'].iloc[0]

    # Format the response
    return {
        "message": f"Is {student_name} AI/ML ready? {bool(prediction)}",
        "ai_ml_ready": bool(prediction),
        "confidence": round(probability, 2),
        "student_name": student_name,
        "skills": data.skills
    }
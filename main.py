from fastapi import FastAPI, UploadFile, File, Form
from pypdf import PdfReader
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import HTTPException
import google.generativeai as genai
import os
import io

app = FastAPI(title="AI Resume Screener API")

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class ScreenerResult(BaseModel):
    candidate_summary: str
    matched_skills: list[str]
    missing_skills: list[str]
    match_score_out_of_100: int
    reasoning: str

@app.get("/")
def read_root():
    return {"status": "AI Resume Screener API is running"}


@app.post("/screen-resume/")
async def screen_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    
    if len(job_description.strip()) < 20:
        raise HTTPException(status_code=400, detail="Job description is too short or empty")

    if resume.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="File must be in PDF format!")
    
    # Read the raw bytes from the uploaded file
    pdf_bytes = await resume.read()

    # Wrap bytes in an in-memory buffer so pypdf can treat it like a file
    pdf_stream = io.BytesIO(pdf_bytes)
    try:
    # Create a PdfReader from that stream
        reader = PdfReader(pdf_stream)

    # Loop through reader.pages and pull text from each one, building one big string
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text()

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read pdf: {str(e)}")

    # Build the prompt combining resume text and job description
    prompt = f"""
    You are an expert resume screener. Evaluate the candidate's resume against
    the job description below. Use only information present in the supplied
    text; do not infer experience, skills, or qualifications that are not stated.

    Return a concise, evidence-based assessment. The match score must be an
    integer from 0 to 100 and should reflect the candidate's fit for the role,
    including required skills, relevant experience, and responsibilities.
    List only skills clearly present in the resume as matched skills, and list
    required job skills absent from the resume as missing skills.

    JOB DESCRIPTION:
    {job_description}

    RESUME TEXT:
    {extracted_text}
    """

    # Create model instance
    model = genai.GenerativeModel("gemini-3.5-flash")

    # Call generate_content, passing generation_config with response_schema
    response = model.generate_content(prompt, 
        generation_config=genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=ScreenerResult))

    # Gemini returns the JSON as a string in response.text
    result = ScreenerResult.model_validate_json(response.text)

    return result
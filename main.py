from fastapi import FastAPI, UploadFile, File, Form
from pypdf import PdfReader
from pydantic import BaseModel
from dotenv import load_dotenv
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
    # 1. Read the raw bytes from the uploaded file (this is async — don't forget `await`)
    pdf_bytes = await resume.read()

    # 2. Wrap those bytes in an in-memory buffer so pypdf can treat it like a file
    pdf_stream = io.BytesIO(pdf_bytes)

    # 3. Create a PdfReader from that stream
    reader = PdfReader(pdf_stream)

    # 4. Loop through reader.pages and pull text from each one, building one big string
    extracted_text = ""
    for page in reader.pages:
        extracted_text += page.extract_text()

    # 5. Build the prompt combining resume text and job description
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

    # 7. Create the Gemini model instance
    model = genai.GenerativeModel("gemini-3.5-flash")

    # 8. Call generate_content, passing generation_config with response_schema
    response = model.generate_content(prompt, 
        generation_config=genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=ScreenerResult))

    # 9. Gemini returns the JSON as a string in response.text — parse it into your Pydantic model
    result = ScreenerResult.model_validate_json(response.text)

    return result
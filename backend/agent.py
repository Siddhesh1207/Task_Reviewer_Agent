import os
import google.generativeai as genai
import logging
import hashlib
from datetime import datetime
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pymongo import MongoClient
from pydantic import BaseModel, Field
from bson import ObjectId
from typing import Any, List
from pydantic_core import core_schema
from dotenv import load_dotenv
import requests
import numpy as np
from passlib.context import CryptContext

# --- Config & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- DB Connection ---
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    MONGO_URI = "mongodb://localhost:27017/"

try:
    client_mongo = MongoClient(MONGO_URI)
    # Send a ping to confirm a successful connection
    client_mongo.admin.command('ping')
    logging.info("Successfully connected to MongoDB!")
except Exception as e:
    logging.error(f"MongoDB Connection Failed: {e}")

db = client_mongo["TaskLoop_db_v2"]
tasks_collection = db["tasks"]
reviews_collection = db["reviews"]
teachers_collection = db["teachers"]

# --- Security ---
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
API_KEY = os.environ.get("AGENT_API_KEY", "12345678")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Invalid API Key")

# --- LLM Setup ---
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    logging.warning("GOOGLE_API_KEY not set.")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# --- Data Models ---
class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, _s, _h) -> core_schema.CoreSchema:
        def v(v: Any) -> ObjectId:
            if not ObjectId.is_valid(v): raise ValueError("Invalid ObjectId")
            return ObjectId(v)
        return core_schema.json_or_python_schema(
            python_schema=core_schema.with_info_plain_validator_function(v),
            json_schema=core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )

class TeacherAuth(BaseModel):
    username: str
    password: str

class Task(BaseModel):
    id: PyObjectId | None = Field(default=None, alias="_id")
    task_id: str
    title: str
    description: str
    teacher_username: str
    class Config: populate_by_name = True; arbitrary_types_allowed = True; json_encoders = {ObjectId: str}

class ReviewSubmission(BaseModel): submission_text: str

class DHIScores(BaseModel):
    dignity: int = Field(..., ge=1, le=10)
    honesty: int = Field(..., ge=1, le=10)
    integrity: int = Field(..., ge=1, le=10)

class ReviewData(BaseModel):
    score: int
    done_well: List[str]
    missing: List[str]
    submission_summary: str = ""

class NextTask(BaseModel):
    title: str; objectives: List[str]; deliverables: str

class Feedback(BaseModel):
    sentiment: str
    dhi_scores: DHIScores

class ReviewHistory(BaseModel):
    id: PyObjectId | None = Field(default=None, alias="_id")
    review_id: str = Field(default_factory=lambda: str(ObjectId()))
    student_username: str
    teacher_username: str
    task_id: str
    review_data: ReviewData
    feedback_note: str
    next_task: NextTask
    feedback_sentiment: str | None = None
    dhi_scores: DHIScores | None = None
    overall_score: float | None = None
    status: str = Field(default="pending_feedback")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    class Config: populate_by_name = True; arbitrary_types_allowed = True; json_encoders = {ObjectId: str}

# --- Prompts ---
review_parser = PydanticOutputParser(pydantic_object=ReviewData)
review_chain = PromptTemplate.from_template(
    """
    ROLE: Expert Code Reviewer.
    TASK: Review submission against: {task_description}
    SUBMISSION: {submission_text}
    {format_instructions}
    """
, partial_variables={"format_instructions": review_parser.get_format_instructions()}) | model | review_parser

note_chain = PromptTemplate.from_template(
    """Write a 2-sentence encouraging feedback note for a student who got {score}/10. 
    Mention: {done_well} and {missing}."""
) | model | StrOutputParser()

next_task_parser = PydanticOutputParser(pydantic_object=NextTask)
next_task_chain = PromptTemplate.from_template(
    """
    ROLE: Coding Mentor.
    CONTEXT: Student scored {score}/10. Weaknesses: {missing}.
    TASK: Create a follow-up coding task to fix these weaknesses.
    {format_instructions}
    """
, partial_variables={"format_instructions": next_task_parser.get_format_instructions()}) | model | next_task_parser

# --- API ---
app = FastAPI(title="TaskLoop", version="2.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health_check():
    return {"status": "running"}

# --- AUTH ROUTES ---
@app.post("/auth/teacher/register", tags=["Auth"])
def register_teacher(auth: TeacherAuth):
    try:
        if teachers_collection.find_one({"username": auth.username}):
            raise HTTPException(status_code=400, detail="Username already exists")

        # Simple, direct hashing (No length limit!)
        hashed_pw = pwd_context.hash(auth.password)

        teachers_collection.insert_one({"username": auth.username, "password": hashed_pw})
        return {"status": "success", "username": auth.username}
    except Exception as e:
        logging.error(f"Register Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/teacher/login", tags=["Auth"])
def login_teacher(auth: TeacherAuth):
    try:
        user = teachers_collection.find_one({"username": auth.username})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Direct verification
        if not pwd_context.verify(auth.password, user["password"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        return {"status": "success", "username": auth.username}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Login Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/auth/check-teacher/{username}", tags=["Auth"])
def check_teacher_exists(username: str):
    if not teachers_collection.find_one({"username": username}):
        raise HTTPException(status_code=404, detail="Teacher not found")
    return {"status": "exists", "username": username}

# --- TASK ROUTES ---
@app.post("/tasks", dependencies=[Depends(get_api_key)], tags=["Tasks"])
def create_task(task: Task):
    if tasks_collection.find_one({"task_id": task.task_id, "teacher_username": task.teacher_username}):
        raise HTTPException(status_code=400, detail="Task ID already exists for this teacher")
    tasks_collection.insert_one(task.model_dump(by_alias=True, exclude=["id"]))
    return {"status": "success", "task_id": task.task_id}

@app.get("/tasks/{teacher_username}", dependencies=[Depends(get_api_key)], tags=["Tasks"])
def get_teacher_tasks(teacher_username: str):
    return list(tasks_collection.find({"teacher_username": teacher_username}, {"_id": 0}))

# --- REVIEW LOGIC ---
def _process_review(teacher_username: str, student_username: str, task_id: str, submission_text: str):
    task = tasks_collection.find_one({"teacher_username": teacher_username, "task_id": task_id})
    if not task: raise HTTPException(404, "Task not found")
    
    try:
        review_data = review_chain.invoke({"task_description": task['description'], "submission_text": submission_text})
        note = note_chain.invoke(review_data.model_dump())
    except Exception as e:
        logging.error(f"AI Error: {e}")
        raise HTTPException(500, f"AI Error: {str(e)}")

    history = ReviewHistory(
        teacher_username=teacher_username,
        student_username=student_username,
        task_id=task_id,
        review_data=review_data,
        feedback_note=note,
        next_task=NextTask(title="", objectives=[], deliverables="")
    )
    res = reviews_collection.insert_one(history.model_dump(by_alias=True, exclude=["id"]))
    return {**history.model_dump(by_alias=True, exclude=["id"]), "review_id": str(history.review_id)}

@app.post("/review/text/{teacher_username}/{student_username}/{task_id}", dependencies=[Depends(get_api_key)])
def review_text(teacher_username: str, student_username: str, task_id: str, sub: ReviewSubmission):
    return _process_review(teacher_username, student_username, task_id, sub.submission_text)

# --- FEEDBACK & HISTORY ---
@app.get("/reviews/teacher/{teacher_username}", dependencies=[Depends(get_api_key)])
def get_reviews_for_teacher(teacher_username: str):
    reviews = list(reviews_collection.find({"teacher_username": teacher_username, "status": "pending_feedback"}))
    for r in reviews: r["_id"] = str(r["_id"])
    return reviews

@app.get("/reviews/student/{teacher_username}/{student_username}", dependencies=[Depends(get_api_key)])
def get_reviews_for_student(teacher_username: str, student_username: str):
    reviews = list(reviews_collection.find({"teacher_username": teacher_username, "student_username": student_username}))
    for r in reviews: r["_id"] = str(r["_id"])
    return reviews

@app.post("/admin/feedback/{review_id}", dependencies=[Depends(get_api_key)])
def submit_admin_feedback(review_id: str, feedback: Feedback):
    review = reviews_collection.find_one({"review_id": review_id})
    if not review: raise HTTPException(404, "Review not found")
    
    ai_score = review['review_data']['score']
    dhi = feedback.dhi_scores
    overall = round(np.mean([ai_score, dhi.dignity, dhi.honesty, dhi.integrity]), 2)
    
    reviews_collection.update_one({"review_id": review_id}, {
        "$set": {
            "dhi_scores": dhi.model_dump(),
            "overall_score": overall,
            "status": "feedback_provided",
            "feedback_sentiment": feedback.sentiment
        }
    })
    return {"status": "success"}

@app.post("/student/generate-next/{review_id}", dependencies=[Depends(get_api_key)])
def generate_next(review_id: str):
    review = reviews_collection.find_one({"review_id": review_id})
    if not review or review['status'] != 'feedback_provided':
        raise HTTPException(400, "Feedback not ready")
        
    next_task = next_task_chain.invoke(review['review_data'])
    reviews_collection.update_one({"review_id": review_id}, {"$set": {"next_task": next_task.model_dump()}})
    
    updated = reviews_collection.find_one({"review_id": review_id})
    updated["_id"] = str(updated["_id"])
    return updated

@app.get("/review/{review_id}", dependencies=[Depends(get_api_key)], tags=["Reviews"])
def get_review_detail(review_id: str):
    review = reviews_collection.find_one({"review_id": review_id})
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    review["_id"] = str(review["_id"])
    return review
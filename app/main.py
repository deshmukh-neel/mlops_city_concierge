import os
from pathlib import Path
from fastapi import Depends, FastAPI
from fastapi import HTTPException
from psycopg2.extensions import connection
from .db import get_db
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional


app = FastAPI()
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

# Request model for custom recommendations
class RecommendationRequest(BaseModel):
    query: str = Field(..., description="User's recommendation query (e.g., 'Find wine bars with outdoor seating')")
    limit: int = Field(default=5, ge=1, le=20, description="Number of recommendations to return")

# Response model for a single place
# will be used for frontend integration later
class Place(BaseModel):
    place_id: str
    name: str
    primary_type: str
    rating: Optional[float] = None
    user_rating_count: Optional[int] = None
    price_level: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None
    
    class Config:
        from_attributes = True


def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY in environment.")
    return genai.Client(api_key=api_key)

@app.get("/root")
def root() -> dict[str, str]:
    return {"message": "Welcome!"}

@app.get("/health")
def health() -> dict[str, str]:
    _ = get_gemini_client()
    return {"status": "Successfully created Gemini client"}


@app.get("/health/db")
def health_db(conn: connection = Depends(get_db)) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
        _ = cur.fetchone()
    return {"status": "ok"}

@app.get("/predict")
def predict() -> dict[str, str]:
    client = get_gemini_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Find me a list of wine bars in North Beach",
    )
    return {"response": response.text or ""}

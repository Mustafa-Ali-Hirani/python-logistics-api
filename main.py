import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

# 1. INITIALIZE FASTAPI (Exactly like 'const app = express()' in Node.js)
app = FastAPI(title="AI Logistics Microservice")

# 2. DEFINE THE REQUEST BODY SCHEMA (Like a Mongoose Schema or Zod validator)
class AuditRequest(BaseModel):
    cargo_description: str

# 3. DEFINE THE API ENDPOINT (Exactly like 'app.post("/api/audit", ...)' in Express)
@app.post("/api/audit")
def audit_cargo_endpoint(request: AuditRequest):
    try:
        # A. Run Agent 1 (Planner)
        planner_config = types.GenerateContentConfig(
            system_instruction="You are a Logistics Planner. Create a fast shipping plan."
        )
        planner_response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Create a shipping route for: {request.cargo_description}",
            config=planner_config
        )
        proposed_plan = planner_response.text

        # B. Run Agent 2 (Safety Officer)
        safety_config = types.GenerateContentConfig(
            system_instruction="""You are a strict Safety Compliance Officer. 
            RULE: Chemical Fuel CANNOT be shipped via Air transport. 
            End review with 'DECISION: APPROVED' or 'DECISION: REJECTED'."""
        )
        safety_response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Review this plan for safety: {proposed_plan}",
            config=safety_config
        )
        audit_result = safety_response.text

        # C. RETURN THE JSON RESPONSE (Like 'res.json({...})' in Express)
        return {
            "status": "success",
            "proposed_route": proposed_plan,
            "safety_audit": audit_result,
            "is_approved": "APPROVED" in audit_result
        }

    except Exception as e:
        # Catch errors and send a 500 error (Like 'res.status(500).send(err)')
        raise HTTPException(status_code=500, detail=str(e))

# To test if the server is alive
@app.get("/")
def read_root():
    return {"message": "AI Logistics API is running!"}
import os
import json
from typing import List, Optional
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

# --- 1. SETUP ---
load_dotenv() # In JS, this is 'require("dotenv").config()'
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- 2. THE PRODUCTION-GRADE SCHEMA ---
# Think of this as a TypeScript Interface + Zod Validation combined.
class ShipmentAudit(BaseModel):
    # 'Field' lets us set business rules like minimum/maximum values
    shipper: str = Field(..., description="Company name")
    container_id: str = Field(..., description="Unique ID")
    
    # Validation: weight MUST be between 0 and 30,000. 
    # This is much cleaner than writing multiple if/else statements in JS.
    weight_kg: float = Field(..., gt=0, lt=30000)
    
    is_fragile: bool = Field(default=False)
    
    # Pattern: This is a Regex that forces the AI to pick one of three choices.
    priority_level: str = Field(..., pattern="^(Low|Medium|High)$")
    
    audit_remarks: str = Field(..., description="One sentence summary")

# --- 3. THE AUDITOR FUNCTION ---
def audit_shipment(raw_input: str):
    print(f"\n[STEP] Auditing Input: {raw_input[:60]}...")
    
    # We use a 'Triple Quote' string for the prompt to keep it clean.
    system_instruction = """
    You are a Logistics Auditor. Extract data into a flat JSON object.
    MANDATORY KEYS: shipper, container_id, weight_kg, is_fragile, priority_level, audit_remarks.
    RULES: 
    - weight_kg must be a number.
    - priority_level must be 'Low', 'Medium', or 'High'.
    - Return ONLY the JSON object, no intro text.
    """

    try:
        # Call the LLM (AI Brain)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": raw_input}
            ],
            response_format={"type": "json_object"}
        )

        # Get raw text and convert to Python Dictionary
        raw_json_str = completion.choices[0].message.content
        data_from_ai = json.loads(raw_json_str)

        # VALIDATION: This is the 'moment of truth'
        # Pydantic checks if the keys match and if weight/priority follow the rules.
        report = ShipmentAudit(**data_from_ai)

        print("✅ SUCCESS: Shipment validated and passed audit.")
        print(report.model_dump_json(indent=2))

    except Exception as error:
        # If the AI gives bad data or misses a key, this block catches it.
        # This is exactly like a 'try/catch' error handler in a Node.js Express route.
        print(f"❌ AUDIT REJECTED: Business rule violation!")
        print(f"Error Details: {error}")

# --- 4. TEST SUITE ---
if __name__ == "__main__":
    # TEST 1: This follows all rules.
    print("\n--- RUNNING TEST 1 (VALID DATA) ---")
    audit_shipment("Shipper: FastCargo. ID: FC-99. Weight: 1500kg. Priority: High. No glass inside.")

    # TEST 2: This violates the 'lt=30000' (less than 30k) rule.
    print("\n--- RUNNING TEST 2 (INVALID WEIGHT) ---")
    audit_shipment("Shipper: Titan Logistics. ID: TL-01. Weight: 55,000kg. Priority: Low.")
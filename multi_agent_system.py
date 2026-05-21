import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. SETUP - Load env and initialize the Google Client
load_dotenv()
# The new SDK automatically looks for 'GEMINI_API_KEY' in your .env!
client = genai.Client()

# --- 2. THE MULTI-AGENT WORKFLOW ---
def run_logistics_team(cargo_description: str):
    print(f"\n[CLIENT CARGO REQUEST]: {cargo_description}")
    
    # STEP A: Planner Agent drafts the route
    print("\n--- 📝 [Agent 1: Planner] is drafting a route... ---")
    
    planner_config = types.GenerateContentConfig(
        system_instruction="""You are a Logistics Planner. 
        Your job is to create a fast and detailed shipping route plan. 
        Include: Estimated time, transport types (Air, Sea, or Road), and stops."""
    )
    
    # We use Google's default fast model: 'gemini-2.5-flash'
    planner_response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"Create a shipping route for: {cargo_description}",
        config=planner_config
    )
    proposed_plan = planner_response.text
    print(proposed_plan)
    
    # STEP B: Safety Agent reviews the Planner's plan
    print("\n--- 🛡️ [Agent 2: Safety Officer] is auditing the plan... ---")
    
    safety_config = types.GenerateContentConfig(
        system_instruction="""You are a strict Safety Compliance Officer. 
        Review the proposed Shipping Plan.
        RULE: Chemical Fuel CANNOT be shipped via Air transport due to explosion hazards.
        You must end your review with either 'DECISION: APPROVED' or 'DECISION: REJECTED' in bold."""
    )
    
    safety_response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"Review this plan for hazard safety: {proposed_plan}",
        config=safety_config
    )
    print(safety_response.text)

# --- TEST THE TEAM ---
if __name__ == "__main__":
    cargo = "500 Liters of Chemical Fuel from Shanghai to Berlin. Needs to get there as fast as possible."
    run_logistics_team(cargo)
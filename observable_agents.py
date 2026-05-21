import os
import json
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

# --- 1. THE OBSERVABILITY LOGGER ---
# This class acts as our "Black Box" flight recorder.
class AgentTracker:
    def __init__(self):
        self.trace = {
            "session_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "timestamp": str(datetime.now()),
            "steps": []
        }

    def log_step(self, agent_name: str, input_data: str, output_data: str):
        """Records the exact input and output of an agent step."""
        step = {
            "step_number": len(self.trace["steps"]) + 1,
            "agent": agent_name,
            "timestamp": str(datetime.now()),
            "input_received": input_data,
            "output_generated": output_data
        }
        self.trace["steps"].append(step)
        print(f"📊 [OBSERVABILITY]: Logged step {step['step_number']} for {agent_name}")

    def save_trace_report(self):
        """Saves the final trace to a JSON file for human audit."""
        filename = f"trace_{self.trace['session_id']}.json"
        with open(filename, "w") as f:
            json.dump(self.trace, f, indent=2)
        print(f"\n📂 [OBSERVABILITY REPORT SAVED]: {filename}")


# --- 2. THE RUNNER ---
def run_monitored_team(cargo_description: str):
    # Initialize our observer
    tracker = AgentTracker()
    
    # STEP A: Planner Agent
    planner_config = types.GenerateContentConfig(
        system_instruction="You are a Logistics Planner. Create a fast shipping plan."
    )
    
    planner_prompt = f"Create a shipping route for: {cargo_description}"
    
    print("\n--- Running Agent 1 (Planner)... ---")
    planner_response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=planner_prompt,
        config=planner_config
    )
    proposed_plan = planner_response.text
    
    # LOG STEP A
    tracker.log_step(
        agent_name="Logistics Planner",
        input_data=planner_prompt,
        output_data=proposed_plan
    )

    # STEP B: Safety Agent
    safety_config = types.GenerateContentConfig(
        system_instruction="""You are a strict Safety Compliance Officer. 
        RULE: Chemical Fuel CANNOT be shipped via Air transport. 
        End review with 'DECISION: APPROVED' or 'DECISION: REJECTED'."""
    )
    
    safety_prompt = f"Review this plan for safety: {proposed_plan}"
    
    print("\n--- Running Agent 2 (Safety Officer)... ---")
    safety_response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=safety_prompt,
        config=safety_config
    )
    audit_result = safety_response.text
    
    # LOG STEP B
    tracker.log_step(
        agent_name="Safety Compliance Officer",
        input_data=safety_prompt,
        output_data=audit_result
    )

    # STEP C: Save the final trace report
    tracker.save_trace_report()

# --- RUN ---
if __name__ == "__main__":
    cargo = "500 Liters of Chemical Fuel from Shanghai to Berlin. Needs to get there as fast as possible."
    run_monitored_team(cargo)
import os
import json
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- 1. THE SHARED MEMORY (State) ---
# In LangGraph, 'State' is like a global object that all nodes can read/write.
class AgentState(TypedDict):
    raw_text: str
    parsed_data: dict
    errors: List[str]
    iterations: int

# --- 2. THE VALIDATION SCHEMA (From Point 4) ---
class Shipment(BaseModel):
    shipper: str
    weight_kg: float = Field(..., gt=0, lt=1000) # Let's set a strict limit of 1000kg

# --- 3. THE NODES (The Actions) ---

def extraction_node(state: AgentState):
    """AI attempts to parse the text. If there were previous errors, it sees them."""
    print(f"\n[NODE]: Extraction (Attempt {state['iterations'] + 1})")
    
    error_context = f"Previous errors to fix: {state['errors']}" if state['errors'] else ""
    
    prompt = f"""Extract 'shipper' and 'weight_kg' from this text: {state['raw_text']}.
    {error_context}
    Return ONLY JSON."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    new_data = json.loads(response.choices[0].message.content)
    return {"parsed_data": new_data, "iterations": state['iterations'] + 1}

def audit_node(state: AgentState):
    """Checks the data against Pydantic rules."""
    print("[NODE]: Auditing data...")
    errors = []
    try:
        Shipment(**state['parsed_data'])
    except Exception as e:
        errors.append(str(e))
    
    return {"errors": errors}

# --- 4. THE ROUTER (The Decision Edge) ---

def should_continue(state: AgentState):
    """Decides: Do we finish or loop back to fix errors?"""
    if not state['errors']:
        print("✅ Audit Passed! Finishing...")
        return "end"
    elif state['iterations'] >= 3:
        print("❌ Too many failures. Stopping.")
        return "end"
    else:
        print(f"⚠️ Errors found: {state['errors']}. Looping back to fix...")
        return "continue"

# --- 5. BUILDING THE GRAPH ---

workflow = StateGraph(AgentState)

# Add our steps
workflow.add_node("extractor", extraction_node)
workflow.add_node("auditor", audit_node)

# Connect the steps
workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "auditor")

# Add the logic: After auditor, run 'should_continue' to decide where to go
workflow.add_conditional_edges(
    "auditor",
    should_continue,
    {
        "continue": "extractor", # Loop back
        "end": END               # Finish
    }
)

# Compile the app
app = workflow.compile()

# --- 6. RUN THE AGENT ---
if __name__ == "__main__":
    # We provide a weight that is TOO HEAVY (1500kg) to force a loop!
    inputs = {
        "raw_text": "CMA CGM just sent a container weighing 1500kg.",
        "parsed_data": {},
        "errors": [],
        "iterations": 0
    }
    
    result = app.invoke(inputs)
    print("\nFINAL RESULT:")
    print(json.dumps(result['parsed_data'], indent=2))
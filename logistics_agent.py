import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# --- 1. DEFINE THE TOOLS (The Python Functions) ---
# In JS, this is just a standard function. 
# Here, it simulates looking up a price in a database.
def calculate_shipping_cost(origin: str, destination: str, weight: float):
    """Calculates the shipping cost based on weight and distance."""
    print(f"\n[TOOL RUNNING]: Calculating cost for {weight}kg from {origin} to {destination}...")
    
    # Simple logic: $5 per kg + $100 flat fee
    cost = (weight * 5) + 100
    return {"cost": f"${cost}", "currency": "USD", "eta": "5 days"}

# --- 2. DEFINE THE TOOL SCHEMA (How the AI 'sees' the tool) ---
# This is like a JSON schema that tells the AI: 
# "Hey, if you need to calculate costs, use this function and give it these 3 values."
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_shipping_cost",
            "description": "Calculate the shipping price and ETA for a package",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "The starting city"},
                    "destination": {"type": "string", "description": "The destination city"},
                    "weight": {"type": "number", "description": "The weight of the package in kg"}
                },
                "required": ["origin", "destination", "weight"]
            }
        }
    }
]

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_agent(user_prompt):
    print(f"\nUser Question: {user_prompt}")
    
    # --- 3. FIRST AI CALL: The AI decides whether to use a tool ---
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": user_prompt}],
        tools=tools, # We pass the tool list here
        tool_choice="auto"
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # Check if the AI wants to use a tool
    if tool_calls:
        print("AI Decision: 'I need to use a tool to answer this.'")
        
        # --- 4. EXECUTE THE PYTHON CODE ---
        for tool_call in tool_calls:
            # The AI provides the arguments (like origin='Dubai', weight=500)
            args = json.loads(tool_call.function.arguments)
            
            # We actually run the Python function here
            result = calculate_shipping_cost(
                origin=args.get("origin"),
                destination=args.get("destination"),
                weight=args.get("weight")
            )
            
            # --- 5. SECOND AI CALL: AI gives the final human-friendly answer ---
            # We give the AI the 'result' of the function so it can talk to the human.
            final_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user", "content": user_prompt},
                    response_message, # Original AI thought
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    }
                ]
            )
            print(f"Final Agent Answer: {final_response.choices[0].message.content}")
    else:
        # If no tool was needed (e.g. "Hi, how are you?")
        print(f"Final Agent Answer: {response_message.content}")

# --- TEST THE AGENT ---
if __name__ == "__main__":
    # Test 1: A general question (No tool needed)
    run_agent("Hi! Who are you?")
    
    # Test 2: A question that REQUIRES the tool
    run_agent("How much will it cost to send a 200kg crate from Karachi to Dubai?")
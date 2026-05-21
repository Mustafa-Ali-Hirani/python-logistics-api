import os
import json
from dotenv import load_dotenv
from groq import Groq
from mcp_server import MockMCPServer # Importing our 'External' server

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 1. Initialize the MCP Server (The Plug)
mcp_server = MockMCPServer()

def ask_mcp_agent(question: str):
    print(f"\nUser Query: {question}")
    
    # 2. DISCOVERY: The Agent 'discovers' what the server knows
    available_containers = mcp_server.list_all_resources()
    
    # 3. CONTEXT: We provide the AI with the info from the MCP server
    prompt = f"""
    You are an AI Agent connected to an MCP Server.
    The server currently has data for these containers: {available_containers}
    
    If the user asks about a container, I will look it up on the server for you.
    
    USER QUESTION: {question}
    """

    # We check if the question mentions a specific container
    found_id = None
    for cid in available_containers:
        if cid in question:
            found_id = cid
            break

    if found_id:
        # AGENT ACTION: Fetch info via the MCP protocol logic
        server_info = mcp_server.get_container_info(found_id)
        print(f"[MCP CONNECTED]: Retrieved data for {found_id} from Server...")
        
        final_prompt = f"{prompt}\n\nSERVER DATA RETRIEVED: {server_info}\nPlease explain this to the user."
    else:
        final_prompt = f"{prompt}\nNo specific container found. Just answer normally."

    # Generate final answer
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": final_prompt}]
    )
    
    print(f"AGENT RESPONSE: {response.choices[0].message.content}")

# --- TEST ---
if __name__ == "__main__":
    # Test 1: Asking about something the 'Server' knows
    ask_mcp_agent("What is the status of container XYZ-999?")
    
    # Test 2: Asking about something the server DOESN'T know
    ask_mcp_agent("Where is my car?")
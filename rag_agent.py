import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- 1. THE KNOWLEDGE BASE ---
# In a real app, this would be thousands of PDFs. 
# Here, we keep it simple to understand the logic.
def search_knowledge_base(query: str):
    print(f"\n[RAG]: Searching port_rules.txt for information about '{query}'...")
    
    # We read the file and find lines that match the keyword
    with open("port_rules.txt", "r") as f:
        rules = f.readlines()
    
    # Simple search logic (In a professional app, we use 'Vector Search')
    relevant_rules = [r for r in rules if any(word.lower() in r.lower() for word in query.split())]
    
    return "\n".join(relevant_rules) if relevant_rules else "No specific port rules found for this query."

# --- 2. THE RAG AGENT ---
def ask_policy_agent(question: str):
    print(f"\nUser Question: {question}")
    
    # Step A: Retrieve information from our private library
    context = search_knowledge_base(question)
    
    # Step B: Augment the prompt (This is the 'A' in RAG)
    # We give the AI the question AND the secret information we found.
    prompt = f"""
    You are a Logistics Expert. Answer the question using ONLY the following port rules.
    If the answer isn't in the rules, say 'I do not have information on that.'
    
    PORT RULES:
    {context}
    
    QUESTION:
    {question}
    """

    # Step C: Generate the answer
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    print(f"\nFINAL ANSWER:\n{response.choices[0].message.content}")

# --- TEST THE RAG AGENT ---
if __name__ == "__main__":
    # Test 1: Info exists in the file
    ask_policy_agent("Where should I store a fragile container?")
    
    # Test 2: Info does NOT exist
    ask_policy_agent("What is the cost of fuel at the port?")
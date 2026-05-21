# Import libraries (In JS, this is like 'const fs = require("fs")' or 'import')
import os
import json
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

# 1. Load the .env file (Like 'dotenv.config()' in Node.js)
load_dotenv()

# 2. Get the API key from environment variables
api_key = os.getenv("GROQ_API_KEY")

# 3. Define the Schema (Like a TypeScript Interface or Zod Schema)
class Shipment(BaseModel):
    shipper_name: str
    consignee: str
    origin_city: str
    destination_city: str
    weight_kg: float
    container_number: str
    estimated_delivery_date: str

# 4. Initialize the AI Client
client = Groq(api_key=api_key)

def parse_with_ai():
    # Read the messy text file
    with open("messy_shipment.txt", "r") as file:
        messy_text = file.read()

    print("--- Sending data to Groq AI... ---")

    # 5. Call the LLM API (Like an Axios/Fetch call to an AI endpoint)
    # We use a 'System Message' to tell the AI how to behave (its role)
    # We use a 'User Message' to give it the actual task
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system", 
                "content": "You are a logistics data extractor. Return ONLY valid JSON."
            },
            {
                "role": "user", 
                "content": f"Extract shipment data from this text: {messy_text}. Match these fields: shipper_name, consignee, origin_city, destination_city, weight_kg, container_number, estimated_delivery_date."
            }
        ],
        # Tell the AI to respond in JSON mode (critical for agents!)
        response_format={"type": "json_object"}
    )

    # 6. Extract the content and validate with Pydantic
    raw_json = completion.choices[0].message.content
    data_dict = json.loads(raw_json)
    
    # This line ensures the data is perfect (Like Zod's .parse() in JS)
    shipment_data = Shipment(**data_dict)

    # 7. Print and Save
    print("\nSUCCESS! Extracted Data:")
    print(shipment_data.model_dump_json(indent=2))

    with open("ai_parsed_output.json", "w") as f:
        f.write(shipment_data.model_dump_json(indent=2))

# Python's 'entry point' (Like the main() function in other languages)
if __name__ == "__main__":
    parse_with_ai()
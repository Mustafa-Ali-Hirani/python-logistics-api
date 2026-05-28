# cargo_damage_analyzer.py
import os
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Initialize the modern Google GenAI Client
# (It automatically loads GEMINI_API_KEY from your .env file)
client = genai.Client()

IMAGE_PATH = "damaged_cargo.jpg"

# Define the Pydantic schema for structured damage assessment
class DamageAssessmentSchema(BaseModel):
    visual_observations: str = Field(description="Detailed physical description of the damage visible in the photo.")
    damage_type: str = Field(description="Category of damage (e.g. Dent/Structural, Water Damage, Broken Seal, Crushed Packaging).")
    severity_level: str = Field(description="Severity classification: Low, Medium, or High.")
    estimated_probable_cause: str = Field(description="The most logical cause of this damage based on visual indicators.")
    draft_insurance_claim_email: str = Field(description="A formal maritime insurance claim email drafted to the carrier (Atlantic Ocean Lines) demanding survey and reimbursement.")

def analyze_cargo_damage():
    print("====================================================")
    print("       LAUNCHING MULTIMODAL DAMAGE ASSESSMENT       ")
    print("====================================================")
    
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Test image {IMAGE_PATH} not found in workspace.")
        
    # 1. Load the image using Pillow
    print(f"[Vision] Loading local image: '{IMAGE_PATH}'...")
    image = Image.open(IMAGE_PATH)
    
    prompt = """
    You are an expert maritime cargo surveyor and insurance auditor.
    Analyze the provided photo of damaged shipping containers.
    
    Conduct a structured inspection and output your findings. Ensure you draft a formal claim email to the carrier.
    """
    
    print("[Gemini] Sending image and prompt to gemini-2.5-flash...")
    
    # 2. Call the Multimodal Model with the updated 2.5-flash model ID
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DamageAssessmentSchema,
            temperature=0.0
        ),
    )
    
    # 3. Print the structured result
    print("\n=== SUCCESS: MULTIMODAL ASSESSMENT COMPLETE ===")
    print(response.text)
    
    # Optional: Write JSON report to a file
    with open("cargo_damage_report.json", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("\n✓ Saved structured report to 'cargo_damage_report.json'")

if __name__ == "__main__":
    analyze_cargo_damage()
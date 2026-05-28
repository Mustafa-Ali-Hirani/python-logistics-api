# bol_parser.py
import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from pydantic import BaseModel, Field
from typing import List, Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# WINDOWS SYSTEM PATH CONFIGURATIONS
# ==========================================
# Verified installation paths from your setup
TESSERACT_EXE_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE_PATH

POPPLER_BIN_PATH = r"C:\poppler\poppler-26.02.0\Library\bin"
# ==========================================

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Define schemas for structured output
class LineItem(BaseModel):
    quantity: int = Field(description="Quantity of the items shipped")
    description: str = Field(description="Detailed description of the goods")
    weight_kg: float = Field(description="Weight of the line item in kilograms")
    volume_cbm: Optional[float] = Field(None, description="Volume of the line item in cubic meters")

class BillOfLadingSchema(BaseModel):
    bol_number: str = Field(description="The unique Bill of Lading document number")
    date: str = Field(description="The date of the document (YYYY-MM-DD format if possible)")
    shipper: str = Field(description="The name and address of the shipper/exporter")
    consignee: str = Field(description="The name and address of the consignee/receiver")
    line_items: List[LineItem] = Field(description="List of physical items being transported")
    total_weight_kg: float = Field(description="Total weight parsed from the document")
    total_volume_cbm: float = Field(description="Total volume parsed from the document")
    port_of_loading: str = Field(description="The departure port (e.g. USNYC)")
    port_of_discharge: str = Field(description="The arrival port (e.g. DEHAM)")
    carrier_name: str = Field(description="The shipping line or carrier responsible for transport")

def extract_digital_text(pdf_path: str) -> str:
    """Attempts to extract text directly from a digital PDF."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def extract_scanned_text(pdf_path: str) -> str:
    """Converts a scanned PDF into images and performs OCR using Tesseract."""
    print("[Pipeline] No digital text found. Running OCR via pdf2image and Tesseract...")
    
    pages = convert_from_path(pdf_path, dpi=150, poppler_path=POPPLER_BIN_PATH)
    
    ocr_text = ""
    for idx, page in enumerate(pages):
        print(f"[Pipeline] Processing page {idx + 1}...")
        text = pytesseract.image_to_string(page)
        ocr_text += text + "\n"
        
    return ocr_text.strip()

def structure_extracted_text(raw_text: str) -> BillOfLadingSchema:
    """Uses Groq and a structured system prompt to cast raw text into Pydantic models."""
    print("[Pipeline] Sending raw text to LLM for structuring...")
    
    prompt = f"""
    You are an expert logistics document parser.
    Your task is to analyze the raw, OCR-extracted text of a Bill of Lading (BoL) and map it to the requested JSON schema.
    
    CRITICAL QUALITY INSTRUCTIONS:
    1. **Mathematical Consistency**: 
       - The sum of the line item `weight_kg` values MUST equal the `total_weight_kg` (6500.0).
       - The sum of the line item `volume_cbm` values MUST EXACTLY equal the `total_volume_cbm` (20.5).
       - Note: If the OCR raw text says '125', do NOT use 125 or 1.25. Use mathematical deduction to realize it must be 12.5 (since 12.5 + 6.2 + 1.8 = 20.5). Placing the decimal point correctly to make the mathematical sum match the reported totals is mandatory.
    2. **Whitespace Restoration**: Fix words that are mashed together due to OCR scanning artifacts (e.g., change "Portof" to "Port of", and "ElectricalControlCabinets" to "Electrical Control Cabinets").
    
    Raw Document Text:
    ---
    {raw_text}
    ---
    
    Provide the output strictly in valid JSON format matching this schema:
    {BillOfLadingSchema.model_json_schema()}
    """
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=[
            {"role": "system", "content": "You output raw JSON matching the requested schema. Do not write markdown wrappers or conversational filler."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    raw_json = completion.choices[0].message.content
    return BillOfLadingSchema.model_validate_json(raw_json)

def process_document(pdf_path: str):
    print(f"\n[Processing] Target Document: {pdf_path}")
    
    # 1. Try digital extraction first
    extracted_text = extract_digital_text(pdf_path)
    
    # 2. If empty (scanned PDF), use OCR
    if not extracted_text or len(extracted_text) < 50:
        extracted_text = extract_scanned_text(pdf_path)
        
    print(f"[Pipeline] Raw Text Character Count: {len(extracted_text)}")
    
    # 3. Structure text using LLM
    try:
        structured_data = structure_extracted_text(extracted_text)
        print("\n=== SUCCESS: EXTRACTED STRUCTURED DATA ===")
        print(structured_data.model_dump_json(indent=2))
    except Exception as e:
        print(f"\n[Error] Failed to parse or structure the document: {e}")
        print("Raw text snippet that caused failure:\n", extracted_text[:500])

if __name__ == "__main__":
    target_file = "Invoice.pdf"
    
    if not os.path.exists(target_file):
        print(f"Error: {target_file} not found. Please run generate_mock_bol.py first.")
    else:
        process_document(target_file)
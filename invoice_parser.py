# invoice_parser.py
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
TESSERACT_EXE_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE_PATH

POPPLER_BIN_PATH = r"C:\poppler\poppler-26.02.0\Library\bin"
# ==========================================

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Define schemas for Commercial Invoice
class InvoiceItem(BaseModel):
    quantity: int = Field(description="The number of units shipped")
    description: str = Field(description="Detailed description of the product/part")
    unit_price: float = Field(description="The price per single unit")
    total_amount: float = Field(description="The total value for this line item (quantity * unit_price)")

class CommercialInvoiceSchema(BaseModel):
    invoice_number: str = Field(description="The unique invoice identifier (e.g., CI No. xxx)")
    date: str = Field(description="The date of issuance (YYYY-MM-DD format if possible)")
    exporter_shipper: str = Field(description="The name and address of the exporter/shipper")
    consignee: str = Field(description="The name and address of the consignee/importer")
    line_items: List[InvoiceItem] = Field(description="List of physical products invoiced")
    currency: str = Field(description="The currency of the invoice (e.g., USD, EUR, PKR)")
    grand_total: float = Field(description="The total monetary value of the invoice (sum of all line_items total_amounts)")

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

def structure_extracted_text(raw_text: str) -> CommercialInvoiceSchema:
    """Uses Groq and a structured prompt to parse text dynamically into our Invoice Schema."""
    print("[Pipeline] Sending raw text to LLM for structuring...")
    
    # We write instructions using variables and math logic instead of hardcoded numbers
    prompt = f"""
    You are an expert logistics and financial auditor.
    Your task is to analyze the raw text of a Commercial Invoice (CI) and extract its structure.
    
    CRITICAL STRUCTURAL INSTRUCTIONS:
    1. **Mathematical Validation**: 
       - For each line item: `total_amount` must equal `quantity * unit_price`.
       - For the overall document: `grand_total` must equal the sum of all individual `total_amount` values extracted.
       - If there is a decimal misplacement in the raw text due to OCR errors, mathematically correct the decimals so that the math is perfectly consistent.
    2. **Mashed Words**: Fix words joined together without spaces by restoring the natural spacing (e.g., "ClipsalPakistan" -> "Clipsal Pakistan").
    
    Raw Document Text:
    ---
    {raw_text}
    ---
    
    Provide the output strictly in valid JSON format matching this schema:
    {CommercialInvoiceSchema.model_json_schema()}
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
    return CommercialInvoiceSchema.model_validate_json(raw_json)

def process_invoice(pdf_path: str):
    print(f"\n[Processing] Target Invoice: {pdf_path}")
    
    # Try digital extraction first
    extracted_text = extract_digital_text(pdf_path)
    
    # Fallback to OCR if empty
    if not extracted_text or len(extracted_text) < 50:
        extracted_text = extract_scanned_text(pdf_path)
        
    print(f"[Pipeline] Raw Text Character Count: {len(extracted_text)}")
    
    try:
        structured_data = structure_extracted_text(extracted_text)
        print("\n=== SUCCESS: EXTRACTED INVOICE DATA ===")
        print(structured_data.model_dump_json(indent=2))
    except Exception as e:
        print(f"\n[Error] Failed to parse or structure the invoice: {e}")
        print("Raw text snippet that caused failure:\n", extracted_text[:500])

if __name__ == "__main__":
    target_file = "Invoice.pdf"
    
    if not os.path.exists(target_file):
        print(f"Error: {target_file} not found in workspace.")
    else:
        process_invoice(target_file)
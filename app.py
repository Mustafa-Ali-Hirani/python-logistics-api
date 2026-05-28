# app.py
import os
import streamlit as st
import pandas as pd
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

# Page Config for a wide, professional layout
st.set_page_config(page_title="Vanguard AI Logistics Portal", layout="wide")

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

# Define Pydantic Schema for Structured Invoicing
class InvoiceItem(BaseModel):
    quantity: int = Field(description="The number of units shipped")
    description: str = Field(description="Detailed description of the product/part")
    unit_price: float = Field(description="The price per single unit")
    total_amount: float = Field(description="The total value (quantity * unit_price)")

class CommercialInvoiceSchema(BaseModel):
    invoice_number: str = Field(description="The unique invoice identifier (e.g., CI No. xxx)")
    date: str = Field(description="The date of issuance (YYYY-MM-DD)")
    exporter_shipper: str = Field(description="The name and address of the exporter")
    consignee: str = Field(description="The name and address of the consignee")
    line_items: List[InvoiceItem] = Field(description="List of physical products invoiced")
    currency: str = Field(description="The currency of the invoice (e.g., USD, EUR)")
    grand_total: float = Field(description="The total monetary value of the invoice")

# ==========================================
# PROCESSING PIPELINE FUNCTIONS
# ==========================================
def extract_pdf_text(pdf_path: str) -> str:
    """Extracts digital text or falls back to OCR if scanned."""
    text = ""
    # Try PyMuPDF digital extraction first
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
            
    # If empty, run OCR
    if not text.strip() or len(text.strip()) < 50:
        pages = convert_from_path(pdf_path, dpi=150, poppler_path=POPPLER_BIN_PATH)
        ocr_text = ""
        for page in pages:
            ocr_text += pytesseract.image_to_string(page) + "\n"
        return ocr_text.strip()
        
    return text.strip()

def structure_text_with_llm(raw_text: str) -> CommercialInvoiceSchema:
    """Uses Groq to parse text into our validated Pydantic schema."""
    client = get_groq_client()
    
    prompt = f"""
    You are an expert logistics and financial auditor.
    Analyze the raw text of a Commercial Invoice (CI) and extract its structure.
    
    CRITICAL STRUCTURAL INSTRUCTIONS:
    1. **Mathematical Validation**: 
       - For each line item: `total_amount` must equal `quantity * unit_price`.
       - For the overall document: `grand_total` must equal the sum of all individual `total_amount` values extracted.
       - If there is a decimal misplacement due to OCR errors, mathematically correct it.
    2. **Whitespace Restoration**: Fix words joined together without spaces.
    
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
            {"role": "system", "content": "You output raw JSON matching the requested schema. Do not write markdown wrappers."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    return CommercialInvoiceSchema.model_validate_json(completion.choices[0].message.content)

# ==========================================
# STREAMLIT UI LAYOUT
# ==========================================
st.title("🚢 Vanguard Global Freight - AI Document Portal")
st.write("Upload scanned shipping invoices or Bills of Lading to automatically extract, audit, and structure your logistics data in real-time.")
st.markdown("---")

# Create Two Columns: Left for Upload, Right for Results
col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Drag and drop your PDF invoice here", type=["pdf"])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily to process it
        temp_filename = "temp_uploaded_invoice.pdf"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.success("PDF uploaded successfully.")
        
        # Action Button
        parse_button = st.button("⚡ Run Enterprise Parser", type="primary")

with col2:
    st.header("2. Audited Extraction Results")
    
    if uploaded_file is not None and parse_button:
        with st.spinner("Executing hybrid OCR extraction & LLM audit..."):
            try:
                # 1. Extract Text
                raw_text = extract_pdf_text(temp_filename)
                
                # 2. Parse text to schema
                structured_data = structure_text_with_llm(raw_text)
                
                # Clean up temporary file
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                
                # ==========================================
                # DISPLAY USER-FRIENDLY METRIC CARDS
                # ==========================================
                st.subheader("📋 Document Metadata")
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Invoice Number", structured_data.invoice_number)
                m_col2.metric("Issue Date", structured_data.date)
                m_col3.metric("Grand Total", f"{structured_data.currency} {structured_data.grand_total:,.2f}")
                
                # Shipper & Consignee boxes
                s_col1, s_col2 = st.columns(2)
                s_col1.info(f"**Exporter / Shipper:**\n\n{structured_data.exporter_shipper}")
                s_col2.success(f"**Consignee / Importer:**\n\n{structured_data.consignee}")
                
                # ==========================================
                # DISPLAY SPREADSHEET VIEW (Interactive Table)
                # ==========================================
                st.markdown("---")
                st.subheader("📦 Audited Line Items")
                
                # Convert Pydantic line items list to a Pandas DataFrame
                items_list = []
                for item in structured_data.line_items:
                    items_list.append({
                        "Quantity": item.quantity,
                        "Description": item.description,
                        "Unit Price": item.unit_price,
                        "Total Amount": item.total_amount
                    })
                
                df_items = pd.DataFrame(items_list)
                
                # Render interactive table with beautiful formatting
                st.dataframe(
                    df_items.style.format({
                        "Unit Price": "${:,.2f}",
                        "Total Amount": "${:,.2f}"
                    }),
                    use_container_width=True,
                    height=400
                )
                
                # Download Button for Excel/CSV
                csv_data = df_items.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export to CSV",
                    data=csv_data,
                    file_name=f"parsed_invoice_{structured_data.invoice_number}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Failed to process document: {e}")
    else:
        st.info("Upload a PDF document in the left sidebar and click 'Run Enterprise Parser' to begin.")
import os
import json
from typing import List
from pydantic import BaseModel, Field
import google.generativeai as genai

# 1. Google Gemini API Key Setup
# اپنی اصل API Key یہاں لازمی درج کریں
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
genai.configure(api_key=GEMINI_API_KEY)

# 2. Define the Desired JSON Output Structure using Pydantic
class InvoiceItem(BaseModel):
    description: str = Field(description="The name or description of the item or service.")
    quantity: int = Field(description="The quantity of the item purchased.")
    unit_price: float = Field(description="The price per unit of the item.")
    total_amount: float = Field(description="The total price for this item line.")

class InvoiceExtractor(BaseModel):
    invoice_number: str = Field(description="The unique invoice ID or invoice number.")
    vendor_name: str = Field(description="The name of the company or person issuing the invoice.")
    invoice_date: str = Field(description="The date the invoice was issued.")
    items: List[InvoiceItem] = Field(description="List of all items or services listed in the invoice.")
    grand_total: float = Field(description="The total overall amount due on the invoice.")

# 3. Sample Unstructured Text Input (Invoice Data)
unstructured_invoice_text = """
INVOICE
Invoice Number: INV-2026-9876
Date: October 25, 2026
Issued By: Datheon IT Solutions Ltd.

Bill To: Faizan Tahir

Items Delivered:
1. Python AI Training Consultant - Qty: 1 - Unit Price: \$150.00 - Total: \$150.00
2. Cloud Infrastructure Setup (AWS) - Qty: 2 - Unit Price: \$75.50 - Total: \$151.00
3. Generative AI API Credits - Qty: 5 - Unit Price: \$10.00 - Total: \$50.00

Grand Total Amount Due: \$351.00
Thank you for your business!
"""

# 4. Construct a Structured Few-Shot Prompt Template
prompt_template = f"""
You are an expert data extraction AI. Your task is to parse the following unstructured invoice text and extract key details into a strictly valid JSON schema.

Target JSON Schema Structure:
{json.dumps(InvoiceExtractor.model_json_schema(), indent=2)}

Unstructured Input Text:
\"\"\"{unstructured_invoice_text}\"\"\"

Instructions:
- Extract all fields accurately.
- Return ONLY a valid JSON object matching the schema above. Do not include any markdown formatting like ```json, notes, or preamble.
"""

# 5. Initialize Gemini Model and Generate Structured Output
try:
    # Using gemini-1.5-flash as the fast benchmarking model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    print("--- Sending Prompt to Gemini LLM Pipeline ---")
    response = model.generate_content(
        prompt_template,
        generation_config={"response_mime_type": "application/json"} # Forces JSON output
    )
    
    # 6. Parse and Validate Output using Pydantic
    raw_json_output = response.text.strip()
    validated_data = InvoiceExtractor.model_validate_json(raw_json_output)
    
    print("\n--- Successfully Extracted & Validated Structured JSON Data ---")
    print(json.dumps(validated_data.model_dump(), indent=4))
    
    # Save the output to a JSON file for task deliverable
    with open('extracted_invoice.json', 'w') as f:
        json.dump(validated_data.model_dump(), f, indent=4)
    print("\nOutput saved successfully to 'extracted_invoice.json'")

except Exception as e:
    print(f"\nAn error occurred during extraction: {e}")

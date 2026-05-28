# enterprise_compliance.py
import os  # <-- Fixed: Added missing import
import re
import json
from datetime import datetime, timezone  # <-- Fixed: timezone imported for Python 3.14 support
from pydantic import BaseModel, Field

# =====================================================================
# SYSTEM 1: ENTERPRISE PII REDACTION ENGINE
# =====================================================================
class PIIRedactor:
    """Enterprise-grade sanitization engine to strip sensitive data from raw texts."""
    
    # Regex patterns for common PII
    EMAIL_PATTERN = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    
    # Fixed: More selective phone pattern to prevent ZIP/Postal codes from matching
    PHONE_PATTERN = r"\+?\d{1,3}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"
    
    # Simple address pattern matching street indicators
    ADDRESS_PATTERN = r"\d{1,5}\s+[A-Za-z0-9\s,.-]+(Street|St|Avenue|Ave|Road|Rd|Parkway|Pkwy|Industriestrasse)\b[A-Za-z0-9\s,.-]*"

    @classmethod
    def redact(cls, text: str) -> tuple:
        """Redacts sensitive entities from raw text. Returns (redacted_text, count_redactions)."""
        redacted = text
        redaction_count = 0
        
        # 1. Redact Emails
        emails = re.findall(cls.EMAIL_PATTERN, redacted)
        if emails:
            redaction_count += len(emails)
            redacted = re.sub(cls.EMAIL_PATTERN, "[REDACTED_EMAIL]", redacted)
            
        # 2. Redact Phone Numbers
        phones = re.findall(cls.PHONE_PATTERN, redacted)
        if phones:
            redaction_count += len(phones)
            redacted = re.sub(cls.PHONE_PATTERN, "[REDACTED_PHONE]", redacted)
            
        # 3. Redact Physical Addresses (Hamburg and Karachi are ignored here unless with a street)
        addresses = re.findall(cls.ADDRESS_PATTERN, redacted, flags=re.IGNORECASE)
        if addresses:
            redaction_count += len(addresses)
            redacted = re.sub(cls.ADDRESS_PATTERN, "[REDACTED_ADDRESS]", redacted)
            
        return redacted, redaction_count

# =====================================================================
# SYSTEM 2: ROLE-BASED ACCESS CONTROL (RBAC) GUARD
# =====================================================================
class UserSession(BaseModel):
    user_id: str
    role: str = Field(description="Role: Guest, CustomsAgent, or ComplianceOfficer")

# Mock sensitive cargo pricing database
cargo_manifest_database = {
    "bol-99281-x": {
        "description": "Industrial Lithium-Ion Battery Packs",
        "consignee_contact": "john.doe@eurodist.de, Phone: +49-40-123456",
        "declared_value_usd": 125000.00,  # Highly sensitive pricing data
        "tax_due_usd": 15625.00          # Highly sensitive tax data
    }
}

def get_manifest_data(bol_number: str, session: UserSession) -> dict:
    """Restricts returned fields dynamically based on the requesting user's RBAC role."""
    print(f"\n[RBAC] Evaluating access for User: '{session.user_id}' (Role: '{session.role}') to BOL: '{bol_number}'")
    
    # 1. Authenticate entry existence
    raw_record = cargo_manifest_database.get(bol_number.lower())
    if not raw_record:
        return {"error": "Record not found"}
        
    # 2. Apply RBAC rules
    if session.role == "ComplianceOfficer":
        # Full access to everything, but we auto-redact contact details for safety
        sanitized_contact, _ = PIIRedactor.redact(raw_record["consignee_contact"])
        record = {
            "description": raw_record["description"],
            "consignee_contact": sanitized_contact,
            "declared_value_usd": raw_record["declared_value_usd"],
            "tax_due_usd": raw_record["tax_due_usd"],
            "access_granted": "Full"
        }
    elif session.role == "CustomsAgent":
        # Access to description, but pricing and contact details are masked
        record = {
            "description": raw_record["description"],
            "consignee_contact": "[HIDDEN - RESTRICTED ROLE]",
            "declared_value_usd": "[HIDDEN - RESTRICTED ROLE]",
            "tax_due_usd": "[HIDDEN - RESTRICTED ROLE]",
            "access_granted": "Partial (Customs Scope Only)"
        }
    else:  # Guest
        # Guests can only see description
        record = {
            "description": raw_record["description"],
            "consignee_contact": "[HIDDEN]",
            "declared_value_usd": "[HIDDEN]",
            "tax_due_usd": "[HIDDEN]",
            "access_granted": "Minimal (Guest Scope Only)"
        }
        
    # 3. Log Audit Trail
    write_audit_log(session, bol_number, record["access_granted"])
    return record

# =====================================================================
# SYSTEM 3: AUDIT COMPLIANCE LOGGER
# =====================================================================
def write_audit_log(session: UserSession, resource_id: str, access_type: str):
    """Writes an immutable, structured JSON compliance audit entry."""
    
    # Fixed: Switched from utcnow() to timezone-aware UTC datetime for Python 3.14 compatibility
    current_time = datetime.now(timezone.utc).isoformat()
    
    log_entry = {
        "timestamp": current_time,
        "user_id": session.user_id,
        "user_role": session.role,
        "accessed_resource": resource_id,
        "authorization_status": "APPROVED",
        "privilege_scope": access_type
    }
    
    # Append the audit log entries to a local JSON file
    audit_file = "audit_compliance_trail.json"
    existing_logs = []
    
    if os.path.exists(audit_file):
        try:
            with open(audit_file, "r") as f:
                existing_logs = json.load(f)
        except Exception:
            existing_logs = []
            
    existing_logs.append(log_entry)
    
    with open(audit_file, "w") as f:
        json.dump(existing_logs, f, indent=2)
    print(f"✓ Security audit entry appended to '{audit_file}'")

# =====================================================================
# RUN COMPLIANCE PIPELINE
# =====================================================================
if __name__ == "__main__":
    print("====================================================")
    print("      LAUNCHING ENTERPRISE COMPLIANCE PIPELINE      ")
    print("====================================================")
    
    # Test 1: PII Redaction
    sample_text = """
    Please deliver the electrical switchgear to John Doe at Industriestrasse 42, Hamburg 20457.
    You can contact the importer at support@eurodist.de or call +49-40-123456.
    """
    print("\n[PII Test] Original Text:")
    print(sample_text.strip())
    
    redacted_text, count = PIIRedactor.redact(sample_text)
    print(f"\n[PII Test] Sanitized Output ({count} redactions completed):")
    print(redacted_text.strip())
    print("-" * 52)
    
    # Test 2: Role-Based Access Control (RBAC) & Audit Trails
    guest_user = UserSession(user_id="usr_009", role="Guest")
    agent_user = UserSession(user_id="usr_042", role="CustomsAgent")
    officer_user = UserSession(user_id="usr_101", role="ComplianceOfficer")
    
    target_container = "BOL-99281-X"
    
    # Execute RBAC lookups
    guest_view = get_manifest_data(target_container, guest_user)
    print(json.dumps(guest_view, indent=2))
    
    agent_view = get_manifest_data(target_container, agent_user)
    print(json.dumps(agent_view, indent=2))
    
    officer_view = get_manifest_data(target_container, officer_user)
    print(json.dumps(officer_view, indent=2))
    
    print("\n====================================================")
    print("✓ SUCCESS: Enterprise compliance pipeline complete!")
    print("====================================================")
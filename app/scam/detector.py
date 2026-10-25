import re
from datetime import datetime, timezone
from typing import Any

SCAM_INDICATORS = {
    "urgency": {
        "weight": 0.25,
        "patterns": [
            r"\b(urgent|urgently|immediately|immediate action|act now|hurry|asap)\b",
            r"\b(within\s+\d+\s+(hour|minute|day)s?|expires\s+in\s+\d+|final notice|deadline)\b",
            r"\b(account.*(suspend|terminat|block|close|deactivat))\b",
            r"\b(do not hesitate|time sensitive|critical warning)\b",
        ],
        "description": "High urgency or pressure tactics to prevent scrutiny",
    },
    "impersonation": {
        "weight": 0.30,
        "patterns": [
            r"\b(ceo|cfo|executive|director|president|board of directors)\b",
            r"\b(it support|helpdesk|system administrator|security team)\b",
            r"\b(irs|fbi|police|tax department|central bank|fraud department|interpol)\b",
            r"\b(bank of|chase|wells fargo|paypal|amazon|apple support|microsoft support)\b",
        ],
        "description": "Authority or executive impersonation cues",
    },
    "suspicious_payment": {
        "weight": 0.35,
        "patterns": [
            r"\b(wire transfer|wire payment|swift|remittance|western union|moneygram)\b",
            r"\b(gift card|apple card|steam card|google play card|vanilla visa)\b",
            r"\b(bitcoin|crypto|usdt|eth|btc wallet|crypto exchange)\b",
            r"\b(new bank details|updated routing number|change invoice account)\b",
            r"\b(overdue payment|unpaid invoice|vendor payment)\b",
        ],
        "description": "Irreversible or altered payment routing instructions",
    },
    "credential_harvesting": {
        "weight": 0.35,
        "patterns": [
            r"\b(otp|one-time password|verification code|2fa|security pin)\b",
            r"\b(password|passcode|secret key|seed phrase|private key)\b",
            r"\b(social security|ssn|national id|date of birth)\b",
            r"\b(cvv|card number|expiration date|atm pin)\b",
            r"\b(click here to verify|login to confirm|confirm your identity)\b",
        ],
        "description": "Request for sensitive credentials, PINs, or verification tokens",
    },
    "coercion_secrecy": {
        "weight": 0.20,
        "patterns": [
            r"\b(keep this confidential|do not tell|keep this between us|strictly confidential)\b",
            r"\b(bypass normal|skip approval|special exception|confidential acquisition)\b",
            r"\b(legal action|arrest warrant|lawsuit|penalty will apply)\b",
        ],
        "description": "Secrecy or bypass of normal accounting/approval protocols",
    },
}

SAMPLE_SCENARIOS = {
    "ceo_fraud_email": (
        "Subject: URGENT: Confidential Vendor Wire Payment Required Today\n\n"
        "Hi, I am currently in a board meeting and cannot take calls. We have a time-sensitive acquisition closing today. "
        "Please execute an urgent wire transfer of $84,500 to our external advisor's account immediately. "
        "The updated routing number is attached. Do not mention this to anyone on the team until the public announcement tomorrow. "
        "Confirm once the wire is completed ASAP.\n\nBest regards,\nMark Sterling - Chief Executive Officer"
    ),
    "fake_tech_support": (
        "CRITICAL SECURITY ALERT from Microsoft Support & IT Helpdesk:\n\n"
        "Your workstation has been compromised with Trojan malware. Your network access will be suspended within 1 hour. "
        "To prevent immediate account termination, share your 6-digit MFA verification code and corporate password with the technician "
        "calling you now. Do not restart your computer or inform local IT."
    ),
    "irs_gift_card_scam": (
        "This is Officer David Miller from the Tax Department and Federal Investigation Bureau. "
        "A formal arrest warrant has been issued in your name for delinquent taxes. "
        "You must immediately purchase $2,500 in Apple gift cards or Vanilla Visa cards and read the codes over the phone to settle this penalty. "
        "If you hang up, police will be dispatched to your location."
    ),
    "legitimate_billing_email": (
        "Subject: Monthly Cloud Infrastructure Invoice - October 2026\n\n"
        "Dear Finance Team,\n\n"
        "Please find attached our regular scheduled monthly service invoice #INV-2026-948 for cloud hosting services. "
        "As usual, this invoice will be processed according to our standard Net-30 payment terms via our existing ACH arrangement. "
        "If you have any questions regarding the line items, feel free to schedule a review with our account manager.\n\nThank you for your business."
    ),
}

class ScamAnalyzer:
    def __init__(self):
        self.indicators = SCAM_INDICATORS

    def analyze(self, text: str, source_type: str = "email") -> dict[str, Any]:
        cleaned_text = str(text).strip()
        if not cleaned_text:
            return {
                "source_type": source_type,
                "risk_score": 0.0,
                "risk_level": "low",
                "flagged_categories": [],
                "evidence": [],
                "explanation": "Empty text provided.",
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
            }

        total_weight = 0.0
        score = 0.0
        flagged_categories = []
        evidence_list = []

        for category, config in self.indicators.items():
            patterns = config["patterns"]
            weight = config["weight"]
            matched_snippets = []

            for pattern in patterns:
                matches = re.finditer(pattern, cleaned_text, re.IGNORECASE)
                for match in matches:
                    snippet = match.group(0).strip()
                    if snippet not in matched_snippets:
                        matched_snippets.append(snippet)

            if matched_snippets:
                score += weight
                flagged_categories.append(category)
                evidence_list.append({
                    "category": category,
                    "description": config["description"],
                    "matched_terms": matched_snippets,
                    "severity_weight": weight,
                })
            total_weight += weight

        # Non-linear scaling: multiple compound flags indicate high coordination
        compound_multiplier = 1.0 + (len(flagged_categories) - 1) * 0.15 if len(flagged_categories) > 1 else 1.0
        final_score = min(1.0, round((score / 1.1) * compound_multiplier, 4))

        if final_score >= 0.70:
            risk_level = "critical_scam"
            explanation = "High-confidence social engineering or scam detected with multiple correlated attack signals."
        elif final_score >= 0.35:
            risk_level = "suspicious"
            explanation = "Elevated risk signals detected. Manual verification via verified out-of-band channels required."
        else:
            risk_level = "low"
            explanation = "No significant scam or social engineering patterns identified."

        return {
            "source_type": source_type,
            "text_length": len(cleaned_text),
            "risk_score": final_score,
            "risk_level": risk_level,
            "flagged_categories": flagged_categories,
            "evidence": evidence_list,
            "explanation": explanation,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

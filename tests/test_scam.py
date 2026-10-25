from app.scam.detector import ScamAnalyzer, SAMPLE_SCENARIOS

def test_ceo_fraud_email_triggers_critical_alert():
    analyzer = ScamAnalyzer()
    res = analyzer.analyze(SAMPLE_SCENARIOS["ceo_fraud_email"], source_type="email")
    assert res["risk_level"] in {"suspicious", "critical_scam"}
    assert res["risk_score"] >= 0.70
    assert "suspicious_payment" in res["flagged_categories"]
    assert "impersonation" in res["flagged_categories"]
    assert "urgency" in res["flagged_categories"]

def test_tech_support_scam_flags_credential_harvesting():
    analyzer = ScamAnalyzer()
    res = analyzer.analyze(SAMPLE_SCENARIOS["fake_tech_support"], source_type="voice_transcript")
    assert res["risk_score"] >= 0.60
    assert "credential_harvesting" in res["flagged_categories"]
    assert "urgency" in res["flagged_categories"]

def test_legitimate_billing_email_is_low_risk():
    analyzer = ScamAnalyzer()
    res = analyzer.analyze(SAMPLE_SCENARIOS["legitimate_billing_email"], source_type="email")
    assert res["risk_level"] == "low"
    assert res["risk_score"] < 0.35
    assert len(res["evidence"]) == 0

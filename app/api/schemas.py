from pydantic import BaseModel, Field, ConfigDict
from typing import Any

class Transaction(BaseModel):
    model_config = ConfigDict(extra="allow")
    # The ULB dataset uses anonymized numeric features. Keeping this flexible
    # allows both Time-present Kaggle files and OpenML mirrors without Time.
    Amount: float = Field(ge=0)
    Time: float | None = Field(default=None, ge=0)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    def to_feature_dict(self) -> dict[str, float]:
        data = self.model_dump()
        data.pop("model_config", None)
        return {k: float(v) for k, v in data.items() if v is not None}

class PredictRequest(BaseModel):
    transaction: Transaction

class BatchPredictRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=500)

class Prediction(BaseModel):
    fraud_probability: float
    risk: str
    threshold: float
    model_version: str
    rollout_mode: str
    decision: str
    shadow_challenger_probability: float | None = None
    canary_rolled_back: bool = False
    explanation: str = "Risk estimate from an anonymized benchmark model; not proof of fraud."

# --- Extension Schemas ---

class ReplayStartRequest(BaseModel):
    scenario: str = Field(default="baseline", description="baseline | elevated_fraud | feature_drift | concept_drift")
    speed_tps: float = Field(default=5.0, ge=0.5, le=100.0)
    label_delay_steps: int = Field(default=10, ge=1, le=100)

class RolloutModeRequest(BaseModel):
    mode: str = Field(description="champion | shadow | canary")
    canary_percent: int | None = Field(default=None, ge=0, le=100)

class PromoteRequest(BaseModel):
    candidate_name: str = Field(default="challenger")
    force: bool = Field(default=False)
    reason: str = Field(default="Promotion approved via operations dashboard")

class RollbackRequest(BaseModel):
    reason: str = Field(default="Manual rollback triggered via operations dashboard")

class ScamTextRequest(BaseModel):
    text: str = Field(min_length=1, description="Email text or transcribed speech to evaluate for scam patterns")
    source_type: str = Field(default="email", description="email | voice_transcript | sms")

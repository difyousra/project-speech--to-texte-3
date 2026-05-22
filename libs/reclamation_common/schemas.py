from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class ClassificationResult(BaseModel):
    intent: str
    category: Optional[str] = None
    service: Optional[str] = None
    priority: Literal["low", "medium", "high", "critical"]
    urgency: bool
    theme: Optional[str] = None
    request_type: str
    level1_label: str
    level2_label: Optional[str] = None
    confidence: dict[str, float] = Field(default_factory=dict)
    waiting_days: Optional[int] = None
    waiting_unit: Optional[str] = None
    handling_sla_days: Optional[int] = None
    service_label: Optional[str] = None


class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    emotion: str
    score: float


class AnalyzeResponse(BaseModel):
    text: str
    classification: ClassificationResult
    sentiment: SentimentResult


class TranscribeResult(BaseModel):
    text: str
    language: str
    confidence: float
    is_partial: bool = False
    validated: bool = False
    validation_reason: Optional[str] = None


class PipelineResult(BaseModel):
    session_id: Optional[str] = None
    transcript: str
    language: str
    classification: ClassificationResult
    sentiment: SentimentResult


class WSEvent(BaseModel):
    type: str
    session_id: str
    payload: dict[str, Any] = Field(default_factory=dict)

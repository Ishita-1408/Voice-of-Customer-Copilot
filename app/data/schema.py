"""Data models and schemas for feedback and intelligence artifacts."""
from typing import TypedDict, List, Optional
from dataclasses import dataclass, field


@dataclass
class FeedbackItem:
    """Standardized representation of a single customer feedback record."""
    id: str
    text: str
    source: Optional[str] = None
    created_at: Optional[str] = None
    customer_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

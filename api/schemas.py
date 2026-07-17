"""api/schemas.py — Pydantic models for all API request / response payloads."""

from typing import Optional
from pydantic import BaseModel


class NewThreadResponse(BaseModel):
    thread_id: str


class ThreadSummary(BaseModel):
    thread_id: str
    message_count: int
    last_active: str


class MessageItem(BaseModel):
    role: str       # "user" | "assistant"
    content: str
    timestamp: str


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    thread_id: str
    response: str
    interrupted: bool = False
    interrupt_prompt: Optional[str] = None


class ResumeRequest(BaseModel):
    thread_id: str
    answer: str     # "yes" | "no"

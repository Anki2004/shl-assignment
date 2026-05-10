"""
main.py
-------
FastAPI service exposing:
  GET  /health  → {"status": "ok"}
  POST /chat    → {"reply", "recommendations", "end_of_conversation"}
"""

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from groq import Groq
from dotenv import load_dotenv

from catalog_index import CatalogIndex
from agent import SHLAgent

load_dotenv()

catalog_index: CatalogIndex = None
agent: SHLAgent             = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global catalog_index, agent

    groq_client   = Groq(api_key=os.environ["GROQ_API_KEY"])
    catalog_index = CatalogIndex()
    catalog_index.load()
    agent = SHLAgent(groq_client=groq_client, catalog_index=catalog_index)

    print("SHL Agent ready.")
    yield
    # cleanup (nothing to do)


app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0",
    lifespan=lifespan,
)



class Message(BaseModel):
    role:    Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        if len(v) > 16:
            raise ValueError("Too many messages (max 16 = 8 turns)")
        if v[-1].role != "user":
            raise ValueError("Last message must be from user")
        return v


class Recommendation(BaseModel):
    name:      str
    url:       str
    test_type: str


class ChatResponse(BaseModel):
    reply:               str
    recommendations:     list[Recommendation]
    end_of_conversation: bool



@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    # Convert Pydantic models → plain dicts for agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        result = agent.chat(messages)
    except Exception as e:
        print(f"[API] Agent error: {e}")
        raise HTTPException(status_code=500, detail="Agent error") from e

    return ChatResponse(
        reply               = result.get("reply", ""),
        recommendations     = [Recommendation(**r) for r in result.get("recommendations", [])],
        end_of_conversation = result.get("end_of_conversation", False),
    )
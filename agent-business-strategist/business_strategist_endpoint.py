from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
from pathlib import Path
import os

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import Client, create_client
from pydantic import BaseModel
from dotenv import load_dotenv

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from business_strategist import primary_agent, memory, Mem0Deps, sub_agents

load_dotenv()

security = HTTPBearer()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))


class AgentRequest(BaseModel):
    query: str
    user_id: str
    request_id: str
    session_id: str


class AgentResponse(BaseModel):
    success: bool


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    expected_token = os.getenv("API_BEARER_TOKEN")
    if not expected_token:
        raise HTTPException(status_code=500, detail="API_BEARER_TOKEN not set")
    if credentials.credentials != expected_token:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return True


def _fetch_history(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    try:
        response = (
            supabase.table("messages")
            .select("*")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data[::-1]
    except Exception as exc:  # pragma: no cover - network/database errors
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {exc}")


def _store_message(session_id: str, message_type: str, content: str, data: Optional[Dict[str, Any]] = None) -> None:
    message_obj: Dict[str, Any] = {"type": message_type, "content": content}
    if data:
        message_obj["data"] = data
    try:
        supabase.table("messages").insert({"session_id": session_id, "message": message_obj}).execute()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to store message: {exc}")


@app.post("/api/business-strategist", response_model=AgentResponse)
async def run_agent(request: AgentRequest, authenticated: bool = Depends(verify_token)) -> AgentResponse:
    try:
        conversation_history = _fetch_history(request.session_id)

        messages = []
        for msg in conversation_history:
            data = msg["message"]
            if data["type"] == "human":
                messages.append(ModelRequest(parts=[UserPromptPart(content=data["content"])]))
            else:
                messages.append(ModelResponse(parts=[TextPart(content=data["content"])]))

        _store_message(request.session_id, "human", request.query)

        relevant = memory.search(query=request.query, user_id=request.user_id, limit=3)
        mem_str = "\n".join(f"- {entry['memory']}" for entry in relevant["results"])
        deps = Mem0Deps(memories=mem_str)

        result = await primary_agent.run(request.query, message_history=messages, deps=deps)

        _store_message(request.session_id, "ai", result.data, {"request_id": request.request_id})

        memory_messages = [
            {"role": "user", "content": request.query},
            {"role": "assistant", "content": result.data},
        ]
        memory.add(memory_messages, user_id=request.user_id)

        return AgentResponse(success=True)
    except Exception as exc:
        _store_message(
            request.session_id,
            "ai",
            "I apologize, but I encountered an error processing your request.",
            {"error": str(exc), "request_id": request.request_id},
        )
        return AgentResponse(success=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        for agent in sub_agents.values():
            await stack.enter_async_context(agent.run_mcp_servers())
        yield


if __name__ == "__main__":
    import uvicorn

    with lifespan(app):
        uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)))


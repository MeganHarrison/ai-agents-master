from __future__ import annotations

"""Interactive Business Strategist orchestrator with Mem0 memory."""

import asyncio
import json
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from mem0 import Memory
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Optional Langfuse instrumentation
try:
    from configure_langfuse import configure_langfuse
except ImportError:  # pragma: no cover - optional dependency
    import importlib.util

    module_path = Path(__file__).resolve().parent / "pydantic-ai-langfuse" / "configure_langfuse.py"
    if module_path.is_file():
        spec = importlib.util.spec_from_file_location("configure_langfuse", module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            try:  # pragma: no cover - best effort import
                spec.loader.exec_module(module)  # type: ignore[attr-defined]
                configure_langfuse = module.configure_langfuse  # type: ignore
            except Exception:
                configure_langfuse = lambda: None
        else:
            configure_langfuse = lambda: None
    else:
        configure_langfuse = lambda: None

load_dotenv()

# ---------------------------------------------------------------------------
# Model and memory setup
# ---------------------------------------------------------------------------

def get_model() -> OpenAIModel:
    """Return the configured OpenAI model."""
    llm = os.getenv("MODEL_CHOICE", "gpt-4o-mini")
    base_url = os.getenv("BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("LLM_API_KEY", "")
    return OpenAIModel(llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key))


MEMORY_CONFIG = {
    "llm": {"provider": "openai", "config": {"model": os.getenv("MODEL_CHOICE", "gpt-4o-mini")}},
    "vector_store": {
        "provider": "supabase",
        "config": {
            "connection_string": os.environ.get("DATABASE_URL", ""),
            "collection_name": "memories",
        },
    },
}

memory = Memory.from_config(MEMORY_CONFIG)


@dataclass
class Mem0Deps:
    memories: str


# ---------------------------------------------------------------------------
# MCP server loading
# ---------------------------------------------------------------------------

PROMPTS: Dict[str, str] = {
    "airtable": "You are an Airtable specialist. Help users interact with Airtable databases.",
    "brave": "You are a web search specialist using Brave Search.",
    "filesystem": "You are a filesystem specialist. Manage files and directories.",
    "github": "You are a GitHub specialist. Manage repositories and issues.",
    "slack": "You are a Slack specialist. Work with channels and messages.",
    "firecrawl": "You are a web crawling specialist. Extract data from websites.",
    "notionApi": "You are a Notion specialist. Work with Notion pages and databases.",
}

CONFIG_PATH = Path(__file__).parent / "mcp_config.json"


def _expand_env(env: Dict[str, str]) -> Dict[str, str]:
    return {k: os.path.expandvars(v) for k, v in env.items()}


def _expand_args(args: list[str]) -> list[str]:
    """Expand environment variables in command arguments."""
    expanded = []
    for arg in args:
        if arg.startswith("$"):
            var = arg[1:]
            default = str(Path.cwd()) if var == "LOCAL_FILE_DIR" else ""
            expanded.append(os.getenv(var, default))
        else:
            expanded.append(os.path.expandvars(arg))
    return expanded


def load_subagents() -> Dict[str, Agent]:
    with open(CONFIG_PATH, "r") as fh:
        config = json.load(fh)

    agents: Dict[str, Agent] = {}
    for name, params in config.get("mcpServers", {}).items():
        command = params.get("command")
        if command is None:
            raise ValueError(
                f'Missing "command" for MCP server "{name}" in {CONFIG_PATH}'
            )

        env = _expand_env(params.get("env", {}))
        server = MCPServerStdio(
            command,
            _expand_args(params.get("args", [])),
            env=env,
        )
        prompt = PROMPTS.get(name, f"You are the {name} specialist.")
        agents[name] = Agent(
            get_model(),
            system_prompt=prompt,
            mcp_servers=[server],
            instrument=os.getenv("ENABLE_LANGFUSE", "false").lower() == "true",
        )
    return agents


try:
    sub_agents = load_subagents()
except Exception as e:
    print(f"Error loading subagents: {e}")
    print("Make sure all required environment variables are set in .env file")
    sys.exit(1)

# Primary orchestrator
instrument = os.getenv("ENABLE_LANGFUSE", "false").lower() == "true"
tracer = configure_langfuse() if instrument else None
primary_agent = Agent(
    get_model(),
    system_prompt=(
        "You are a Business Strategist orchestrator. "
        "Delegate tasks to subagents based on the user's request."
    ),
    deps_type=Mem0Deps,
    instrument=instrument,
)


@primary_agent.system_prompt
def add_memories(ctx: RunContext[str]) -> str:
    return f"\nUser Memories:\n{ctx.deps.memories}"


# Dynamically register tools for each subagent
for _name, _agent in sub_agents.items():

    async def _factory(query: str, agent=_agent) -> Dict[str, str]:
        result = await agent.run(query)
        return {"result": result.data}

    primary_agent.tool_plain(_factory, name=f"use_{_name}_agent")


async def run_cli() -> None:
    """Run the orchestrator in an interactive loop."""

    print("Business Strategist Orchestrator")
    print("Enter 'exit' to quit the program.")

    async with AsyncExitStack() as stack:
        for agent in sub_agents.values():
            await stack.enter_async_context(agent.run_mcp_servers())

        console = Console()
        messages = []

        while True:
            user_input = input("\n[You] ")
            if user_input.lower() in {"exit", "quit", "bye", "goodbye"}:
                print("Goodbye!")
                break

            try:
                memories = memory.search(query=user_input, user_id="default_user", limit=3)
                mem_str = "\n".join(f"- {m['memory']}" for m in memories["results"])
                deps = Mem0Deps(memories=mem_str)
                print("\n[Assistant]")
                curr = ""
                with Live("", console=console, vertical_overflow="visible") as live:
                    async with primary_agent.run_stream(
                        user_input, message_history=messages, deps=deps
                    ) as result:
                        async for msg in result.stream_text(delta=True):
                            curr += msg
                            live.update(Markdown(curr))
                messages.extend(result.all_messages())
                memory_messages = [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": curr},
                ]
                memory.add(memory_messages, user_id="default_user")
            except Exception as exc:  # pragma: no cover - runtime errors
                print(f"\n[Error] {exc}")


if __name__ == "__main__":
    asyncio.run(run_cli())

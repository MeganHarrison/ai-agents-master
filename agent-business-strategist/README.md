# Business Strategist Orchestrator

This module unifies the earlier MCP-based agents into a single orchestrator.  
MCP servers are defined in `mcp_config.json` and loaded dynamically so you can
add or remove tools without changing the code. Mem0 provides long‑term memory
backed by Supabase, and Langfuse instrumentation is enabled when the
`ENABLE_LANGFUSE` environment variable is set to `true`.

## Usage

1. Create a `.env` file with your API keys and model configuration. Required
   variables include `LLM_API_KEY`, `MODEL_CHOICE`, `DATABASE_URL`, and any
   keys referenced by `mcp_config.json` (e.g. `AIRTABLE_API_KEY`,
   `BRAVE_API_KEY`, `GITHUB_TOKEN`, `NOTION_TOKEN`, etc.). The optional
   `LOCAL_FILE_DIR` variable points to the directory that the filesystem MCP
   server can access. If omitted, the current working directory is used.
2. Install dependencies and run the interactive CLI:

```bash
pip install -r requirements.txt
python business_strategist.py
```

The CLI lets you chat with the agent locally while it delegates to the MCP tools
defined in `mcp_config.json`.

## API Endpoint

The file `business_strategist_endpoint.py` exposes a FastAPI app with the path
`/api/business-strategist`. Set the same environment variables as the CLI and
run:

```bash
uvicorn business_strategist_endpoint:app --host 0.0.0.0 --port 8001
```

Point your Next.js application to `http://<host>:8001/api/business-strategist`.

## Docker

Build and run the container on Render or Digital Ocean:

```bash
docker build -t business-strategist .
docker run --env-file .env -p 8001:8001 business-strategist
```

The Dockerfile runs the FastAPI endpoint and exposes port `8001`.

## Configuration

See `mcp_config.json` for an example configuration with common MCP servers,
including a Notion integration.
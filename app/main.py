from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import importlib.util
from pathlib import Path


app = FastAPI(title="UBS Challenge API")


@app.get("/")
async def healthcheck():
    return {"status": "ok", "message": "UBS challenge server running"}


@app.post("/ticketing-agent")
async def assign(request: Request, payload: dict):
    # Enforce Content-Type: application/json for requests
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")

    try:
        # Dynamically load module since filename contains a hyphen
        module_path = Path(__file__).with_name("ticketing-agent.py")
        spec = importlib.util.spec_from_file_location("ticketing_agent_module", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load ticketing-agent module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "ticketing_agent"):
            raise RuntimeError("ticketing_agent function not found in ticketing-agent.py")

        handler = getattr(module, "ticketing_agent")
        result = handler(payload)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/trivia")
async def trivia():
    return {"answers": [1, 2, 3, 4]}



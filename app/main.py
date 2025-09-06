from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.ticketing_agent import ticketing_agent
from app.mst_calculation import mst_calculation


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
        result = ticketing_agent(payload)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/trivia")
async def trivia():
    return {"answers": [4,1,2,2,3,4,4,5,4]}


@app.post("/mst-calculation")
async def assign(request: Request, payload: dict):
    # Enforce Content-Type: application/json for requests
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")

    try:
        result = mst_calculation(payload)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
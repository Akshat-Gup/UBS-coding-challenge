from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import JSONResponse

from app.ticketing_agent import ticketing_agent
from app.blankety_simple import blankety_blanks_simple
from app.trading_formula import trading_formula
from app.mst_calculation import mst_calculation
from app.investigate import investigate

app = FastAPI(title="UBS Challenge API")


@app.get("/")
async def healthcheck():
    return {"status": "ok", "message": "UBS challenge server running", "version": "2.0", "timestamp": "2024-01-15"}


@app.get("/test")
async def test_endpoint():
    return {"message": "This is a test endpoint to verify deployment", "working": True}


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

@app.post("/trading-formula")
async def evaluate_trading_formula(request: Request, payload: list = Body(..., embed=False)):
    # Enforce Content-Type: application/json for requests
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")

    try:
        result = trading_formula(payload)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
  

@app.post("/mst-calculation")
async def mst_calculation_endpoint(request: Request, payload: dict):
    # Enforce Content-Type: application/json for requests
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")

    try:
        # Validate payload format
        if not isinstance(payload, dict):
            raise ValueError("Payload must be a list of objects with 'image' field")
        
        for item in payload:
            if not isinstance(item, dict) or "image" not in item:
                raise ValueError("Each item must be a dictionary with an 'image' field containing base64 data")
        
        result = mst_calculation(payload)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
# @app.post("/trading-formula")
# async def evaluate_trading_formula(request: Request, payload: dict):
#     # Enforce Content-Type: application/json for requests
#     content_type = request.headers.get("content-type", "")
#     if not content_type.startswith("application/json"):
#         raise HTTPException(status_code=415, detail="Content-Type must be application/json")

#     try:
#         result = trading_formula(payload)
#         return JSONResponse(content=result)
#     except Exception as exc:
#         raise HTTPException(status_code=400, detail=str(exc))


@app.post("/blankety")
async def blankety_endpoint(request: Request, payload: dict):
    # Enforce Content-Type: application/json for requests
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")

    try:
        result = blankety_blanks_simple(payload)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/investigate")
async def investigate_endpoint(request: Request, payload: dict):
    # Enforce Content-Type: application/json for requests
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")

    try:
        result = investigate(payload)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import JSONResponse
from typing import Union

# Import all endpoint functions
from app.ticketing_agent import ticketing_agent
from app.blankety_simple import blankety_blanks_simple
from app.trading_formula import trading_formula
from app.investigate import investigate
from app.the_ink_archive import the_ink_archive

# Create FastAPI app
app = FastAPI(title="UBS Challenge API", version="3.1")


@app.get("/")
async def healthcheck():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "message": "UBS challenge server running", 
        "version": "3.1",
        "last_updated": "2024-01-16T15:45:00Z",
        "investigate_fix": "Fixed to accept both dict and list payload formats with Union type annotation",
        "endpoints": [
            "GET /",
            "GET /trivia", 
            "POST /ticketing-agent",
            "POST /blankety",
            "POST /trading-formula",
            "POST /investigate",
            "GET /The-Ink-Archive",
            "POST /The-Ink-Archive"
        ]
    }


@app.get("/trivia")
async def trivia():
    """Get trivia answers"""
    return {"answers": [4, 1, 2, 2, 3, 4, 4, 5, 4]}


@app.get("/debug")
async def debug_info():
    """Debug endpoint to verify deployment status"""
    return {
        "deployment_status": "updated",
        "investigate_endpoint": "accepts dict with networks key",
        "test_payload_format": {
            "networks": [
                {
                    "networkId": "example",
                    "network": [
                        {"spy1": "agent1", "spy2": "agent2"}
                    ]
                }
            ]
        }
    }


@app.post("/ticketing-agent")
async def ticketing_agent_endpoint(request: Request, payload: dict):
    """Assign customers to concerts based on ticketing logic"""
    # Enforce Content-Type: application/json for requests
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")

    try:
        result = ticketing_agent(payload)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/blankety")
async def blankety_endpoint(request: Request, payload: dict):
    """Fill in missing values in time series data using interpolation"""
    # Enforce Content-Type: application/json for requests
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")

    try:
        result = blankety_blanks_simple(payload)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/trading-formula")
async def trading_formula_endpoint(request: Request, payload: list = Body(..., embed=False)):
    """Evaluate trading formulas on financial data"""
    # Enforce Content-Type: application/json for requests
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")

    try:
        result = trading_formula(payload)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/investigate")
async def investigate_endpoint(request: Request, payload: Union[dict, list] = Body(...)):
    """Find extra channels in spy networks that can be safely removed"""
    # Enforce Content-Type: application/json for requests
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")

    try:
        # Handle both formats: {"networks": [...]} or direct list
        if isinstance(payload, dict) and "networks" in payload:
            # Already in the correct format
            result = investigate(payload)
        elif isinstance(payload, list):
            # Wrap list in the expected format
            formatted_payload = {"networks": payload}
            result = investigate(formatted_payload)
        else:
            # Handle the case where payload is a dict but doesn't have "networks" key
            # Assume the dict contains the network data directly
            formatted_payload = {"networks": [payload]}
            result = investigate(formatted_payload)
        
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/The-Ink-Archive")
async def the_ink_archive_get():
    """Get sample data for The Ink Archive trading challenge"""
    return {
        "challenge": "The-Ink-Archive",
        "description": "Find optimal trading sequences to maximize gain through bartering cycles",
        "sample_input": {
            "goods": ["Blue Moss", "Amberback Shells", "Kelp Silk", "Ventspice"],
            "rates": [
                [0, 0.9, 0.8, 0.7],
                [1.1, 0, 0.85, 0.75],
                [1.2, 1.15, 0, 0.9],
                [1.3, 1.25, 1.1, 0]
            ]
        },
        "expected_output_format": [
            {
                "path": ["Good1", "Good2", "Good3", "Good1"],
                "gain": 0
            },
            {
                "path": ["Good4", "Good1", "Good2", "Good4"],
                "gain": 1880
            }
        ]
    }


@app.post("/The-Ink-Archive")
async def the_ink_archive_endpoint(request: Request, payload: Union[dict, list] = Body(...)):
    """Find optimal trading sequences to maximize gain through bartering cycles"""
    # Enforce Content-Type: application/json for requests
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")

    try:
        # Handle both possible input formats
        if isinstance(payload, list):
            # If payload is a list, assume it's the rates and create default goods
            processed_payload = {
                "goods": ["Blue Moss", "Amberback Shells", "Kelp Silk", "Ventspice"],
                "rates": payload
            }
        elif isinstance(payload, dict):
            processed_payload = payload
        else:
            raise ValueError("Payload must be a dict with goods/rates or a list of rates")
            
        result = the_ink_archive(processed_payload)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# Optional: Add CORS middleware for browser testing
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add a docs redirect for easier access
@app.get("/docs-redirect")
async def docs_redirect():
    """Redirect to API documentation"""
    return {"message": "Visit /docs for API documentation"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
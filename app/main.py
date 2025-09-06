from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.assignment import assign_concerts


app = FastAPI(title="UBS Challenge API")


@app.get("/")
async def healthcheck():
    return {"status": "ok", "message": "UBS challenge server running"}


@app.post("/assign")
async def assign(payload: dict):
    try:
        result = assign_concerts(payload)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))



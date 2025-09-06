# UBS-coding-challenge
UBS Coding Challenge Solutions

## Local Development (Node.js)

Prereqs: Node.js 18+

Install and run:

```
npm install
npm run dev
```

API:
- GET `/` → health check JSON
- POST `/assign` → accepts JSON payload and returns mapping

Example request body: see `examples/sample-request.json`.

## Deployment (Render)

This repo includes `render.yaml`. To deploy on Render:

1. Push this repo to GitHub.
2. In Render, click "New +" → "Blueprint" and select this repo.
3. Confirm the service created from `render.yaml` (type: web, plan: free).
4. Deploy. Render sets `PORT` automatically; app listens on it.

Alternatively, you can create a Web Service manually:
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health Check Path: `/`

Once deployed, test:

```
curl -s https://<your-service>.onrender.com/
curl -s -X POST -H 'Content-Type: application/json' \
  --data @examples/sample-request.json https://<your-service>.onrender.com/assign

## Local Development (Python)

Prereqs: Python 3.10+

Create venv and run:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Test locally:

```
curl -s http://localhost:8000/
curl -s -X POST -H 'Content-Type: application/json' \
  --data @examples/sample-request.json http://localhost:8000/assign
```
```

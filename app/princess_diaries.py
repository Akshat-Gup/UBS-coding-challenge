from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict, List
import heapq

app = FastAPI()

def calculate_optimized_schedule(data: Dict) -> Dict:
    """
    Optimize task scheduling to maximize score while minimizing transition fees.
    """
    tasks = data['tasks']
    subway = data['subway']
    start_station = data['starting_station']
    
    # Sort tasks by end time
    tasks.sort(key=lambda x: x['end'])
    
    # Create complete station connection graph with fees
    stations = set()
    for task in tasks:
        stations.add(task['station'])
    for conn in subway:
        stations.update(conn['connection'])
    stations.add(start_station)
    
    n = max(stations) + 1
    
    # Initialize distance matrix with Floyd-Warshall
    dist = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    
    # Add direct connections
    for connection in subway:
        i, j = connection['connection']
        fee = connection['fee']
        dist[i][j] = min(dist[i][j], fee)
        dist[j][i] = min(dist[j][i], fee)
    
    # Floyd-Warshall algorithm to find all-pairs shortest path
    for k in range(n):
        for i in range(n):
            if dist[i][k] == float('inf'):
                continue
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    # Dynamic programming
    n_tasks = len(tasks)
    dp = [0] * n_tasks  # max score ending at task i
    prev = [-1] * n_tasks  # previous task index
    total_fees = [0] * n_tasks  # total fees to reach task i
    
    for i in range(n_tasks):
        # Base case: go from start station to task i's station
        dp[i] = tasks[i]['score']
        total_fees[i] = dist[start_station][tasks[i]['station']]
        
        for j in range(i):
            if tasks[j]['end'] <= tasks[i]['start']:  # No time conflict
                # Fee to move from task j's station to task i's station
                move_fee = dist[tasks[j]['station']][tasks[i]['station']]
                new_score = dp[j] + tasks[i]['score']
                new_fee = total_fees[j] + move_fee
                
                # Update if better score or same score with lower fee
                if new_score > dp[i] or (new_score == dp[i] and new_fee < total_fees[i]):
                    dp[i] = new_score
                    prev[i] = j
                    total_fees[i] = new_fee
    
    # Find optimal solution
    max_score = 0
    min_fee = float('inf')
    best_idx = -1
    
    for i in range(n_tasks):
        if dp[i] > max_score or (dp[i] == max_score and total_fees[i] < min_fee):
            max_score = dp[i]
            min_fee = total_fees[i]
            best_idx = i
    
    # Reconstruct schedule
    schedule = []
    current = best_idx
    while current != -1:
        schedule.append(tasks[current]['name'])
        current = prev[current]
    
    schedule.reverse()
    
    return {
        "max_score": max_score,
        "min_fee": min_fee,
        "schedule": schedule
    }

@app.post("/princess-diaries")
async def optimize_schedule_endpoint(request: Request, payload: dict = Body(..., embed=False)):
    """Optimize task scheduling with station transitions"""
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")

    try:
        # Call the internal function
        result = calculate_optimized_schedule(payload)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/")
async def root():
    return {"message": "Princess Diaries Optimization API is running!"}

# For testing locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
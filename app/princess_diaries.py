# def calculate_optimized_schedule(data: dict) -> dict:
#     """
#     Optimize task scheduling including the return trip to starting station
#     """
#     tasks = data['tasks']
#     subway = data['subway']
#     start_station = data['starting_station']
    
#     # Sort tasks by end time
#     tasks.sort(key=lambda x: x['end'])
    
#     # Build station fee graph using Dijkstra
#     def dijkstra_all_pairs(stations, connections, start_station):
#         """Compute shortest path fees from start_station to all stations"""
#         dist = {station: float('inf') for station in stations}
#         dist[start_station] = 0
#         visited = set()
        
#         while len(visited) < len(stations):
#             # Find unvisited node with minimum distance
#             current = None
#             min_dist = float('inf')
#             for station in stations:
#                 if station not in visited and dist[station] < min_dist:
#                     min_dist = dist[station]
#                     current = station
            
#             if current is None:
#                 break
                
#             visited.add(current)
            
#             # Update neighbors
#             for conn in connections:
#                 if current in conn['connection']:
#                     neighbor = conn['connection'][0] if conn['connection'][1] == current else conn['connection'][1]
#                     new_dist = dist[current] + conn['fee']
#                     if new_dist < dist[neighbor]:
#                         dist[neighbor] = new_dist
        
#         return dist
    
#     # Get all stations
#     stations = set()
#     for task in tasks:
#         stations.add(task['station'])
#     for conn in subway:
#         stations.update(conn['connection'])
#     stations.add(start_station)
    
#     # Precompute fees from each station to all other stations
#     station_fees = {}
#     for station in stations:
#         station_fees[station] = dijkstra_all_pairs(stations, subway, station)
    
#     # Dynamic programming
#     n = len(tasks)
#     dp = [0] * n  # max score ending at task i
#     prev = [-1] * n  # previous task index
#     fees = [0] * n  # total fees to reach task i (excluding return trip)
    
#     for i in range(n):
#         # Base case: start from starting station to task i
#         dp[i] = tasks[i]['score']
#         fees[i] = station_fees[start_station][tasks[i]['station']]
        
#         for j in range(i):
#             if tasks[j]['end'] <= tasks[i]['start']:  # No time conflict
#                 # Fee from task j's station to task i's station
#                 move_fee = station_fees[tasks[j]['station']][tasks[i]['station']]
#                 total_score = dp[j] + tasks[i]['score']
#                 total_fee = fees[j] + move_fee
                
#                 # Update if better score or same score with lower fee
#                 if total_score > dp[i] or (total_score == dp[i] and total_fee < fees[i]):
#                     dp[i] = total_score
#                     prev[i] = j
#                     fees[i] = total_fee
    
#     # Find optimal solution and add return trip fee
#     max_score = 0
#     min_total_fee = float('inf')
#     best_idx = -1
    
#     for i in range(n):
#         if dp[i] > 0:  # Only consider valid schedules
#             # Add return trip from last task's station back to start
#             return_fee = station_fees[tasks[i]['station']][start_station]
#             total_fee_with_return = fees[i] + return_fee
            
#             if dp[i] > max_score or (dp[i] == max_score and total_fee_with_return < min_total_fee):
#                 max_score = dp[i]
#                 min_total_fee = total_fee_with_return
#                 best_idx = i
    
#     # Reconstruct schedule
#     schedule = []
#     current = best_idx
#     while current != -1:
#         schedule.append(tasks[current]['name'])
#         current = prev[current]
    
#     schedule.reverse()
    
#     return {
#         "max_score": max_score,
#         "min_fee": min_total_fee,
#         "schedule": schedule
#     }
# print(calculate_optimized_schedule({
#   "tasks": [
#     { "name": "A", "start": 480, "end": 540, "station": 1, "score": 2 },
#     { "name": "B", "start": 600, "end": 660, "station": 2, "score": 1 },
#     { "name": "C", "start": 720, "end": 780, "station": 3, "score": 3 },
#     { "name": "D", "start": 840, "end": 900, "station": 4, "score": 1 },
#     { "name": "E", "start": 960, "end": 1020, "station": 1, "score": 4 },
#     { "name": "F", "start": 530, "end": 590, "station": 2, "score": 1 }
#   ],
#   "subway": [
#     { "connection": [0, 1], "fee": 10 },
#     { "connection": [1, 2], "fee": 10 },
#     { "connection": [2, 3], "fee": 20 },
#     { "connection": [3, 4], "fee": 30 }
#   ],
#   "starting_station": 0
# }))


from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict, List, Tuple
import heapq
import time

app = FastAPI()

def calculate_optimized_schedule(data: Dict) -> Dict:
    """
    Optimized task scheduling with efficient algorithms
    """
    start_time = time.time()
    
    tasks = data['tasks']
    subway = data['subway']
    start_station = data['starting_station']
    
    # Early timeout check
    if time.time() - start_time > 5:
        return {"error": "Timeout during initialization"}
    
    # Sort tasks by end time
    tasks.sort(key=lambda x: x['end'])
    
    # Get all unique stations efficiently
    stations = set()
    stations.add(start_station)
    for task in tasks:
        stations.add(task['station'])
    for conn in subway:
        stations.add(conn['connection'][0])
        stations.add(conn['connection'][1])
    
    stations = sorted(stations)
    n_stations = max(stations) + 1
    
    # OPTIMIZATION 1: Use Dijkstra for each station instead of Floyd-Warshall
    def dijkstra(start: int) -> List[int]:
        """Efficient Dijkstra for single source shortest path"""
        dist = [float('inf')] * n_stations
        dist[start] = 0
        heap = [(0, start)]
        
        while heap:
            current_dist, current_station = heapq.heappop(heap)
            if current_dist > dist[current_station]:
                continue
                
            for conn in subway:
                if current_station in conn['connection']:
                    neighbor = conn['connection'][0] if conn['connection'][1] == current_station else conn['connection'][1]
                    new_dist = current_dist + conn['fee']
                    if new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
                        heapq.heappush(heap, (new_dist, neighbor))
        
        return dist
    
    # Precompute station fees efficiently
    station_fees = {}
    for station in stations:
        if time.time() - start_time > 8:  # Timeout during fee calculation
            return {"error": "Timeout during fee calculation"}
        station_fees[station] = dijkstra(station)
    
    # OPTIMIZATION 2: Early pruning for too many tasks
    n_tasks = len(tasks)
    if n_tasks > 1000:
        # Use greedy approximation for very large inputs
        return greedy_solution(tasks, station_fees, start_station, start_time)
    
    # OPTIMIZATION 3: Efficient DP with binary search for compatible tasks
    dp = [0] * n_tasks
    prev = [-1] * n_tasks
    fees = [0] * n_tasks
    
    # Precompute compatible tasks using binary search
    for i in range(n_tasks):
        if time.time() - start_time > 10:  # Timeout during DP
            return {"error": "Timeout during optimization"}
            
        dp[i] = tasks[i]['score']
        fees[i] = station_fees[start_station][tasks[i]['station']]
        
        # Find last compatible task using binary search
        left, right = 0, i - 1
        best_j = -1
        while left <= right:
            mid = (left + right) // 2
            if tasks[mid]['end'] <= tasks[i]['start']:
                best_j = mid
                left = mid + 1
            else:
                right = mid - 1
        
        # Check all potentially compatible tasks (not all j < i)
        if best_j != -1:
            for j in range(best_j, -1, -1):
                if tasks[j]['end'] > tasks[i]['start']:
                    break
                    
                move_fee = station_fees[tasks[j]['station']][tasks[i]['station']]
                new_score = dp[j] + tasks[i]['score']
                new_fee = fees[j] + move_fee
                
                if new_score > dp[i] or (new_score == dp[i] and new_fee < fees[i]):
                    dp[i] = new_score
                    prev[i] = j
                    fees[i] = new_fee
    
    # Find optimal solution with return trip
    max_score = 0
    min_total_fee = float('inf')
    best_idx = -1
    
    for i in range(n_tasks):
        return_fee = station_fees[tasks[i]['station']][start_station]
        total_fee = fees[i] + return_fee
        
        if dp[i] > max_score or (dp[i] == max_score and total_fee < min_total_fee):
            max_score = dp[i]
            min_total_fee = total_fee
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
        "min_fee": min_total_fee,
        "schedule": schedule
    }

def greedy_solution(tasks: List[Dict], station_fees: Dict, start_station: int, start_time: float) -> Dict:
    """Greedy approximation for very large inputs"""
    tasks.sort(key=lambda x: x['end'])
    
    selected_tasks = []
    current_time = 0
    current_station = start_station
    total_score = 0
    total_fee = 0
    
    for task in tasks:
        if time.time() - start_time > 12:
            break
            
        if task['start'] >= current_time:
            move_fee = station_fees[current_station][task['station']]
            
            selected_tasks.append(task['name'])
            total_score += task['score']
            total_fee += move_fee
            current_time = task['end']
            current_station = task['station']
    
    # Add return trip
    return_fee = station_fees[current_station][start_station]
    total_fee += return_fee
    
    return {
        "max_score": total_score,
        "min_fee": total_fee,
        "schedule": selected_tasks
    }

@app.post("/princess-diaries")
async def optimize_schedule_endpoint(request: Request, payload: dict = Body(..., embed=False)):
    """Optimize task scheduling with station transitions"""
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")

    try:
        result = calculate_optimized_schedule(payload)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/")
async def root():
    return {"message": "Princess Diaries Optimization API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
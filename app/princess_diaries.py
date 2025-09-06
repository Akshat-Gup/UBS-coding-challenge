from typing import Dict, List, Tuple
import heapq
import bisect

def calculate_optimized_schedule(data: Dict) -> Dict:
    """
    Ultra-optimized solution for task scheduling with station transitions
    """
    tasks = data['tasks']
    subway = data['subway']
    start_station = data['starting_station']
    
    # Handle empty tasks case
    if not tasks:
        return {
            "max_score": 0,
            "min_fee": 0,
            "schedule": []
        }
    
    # Sort tasks by end time for DP
    tasks.sort(key=lambda x: x['end'])
    n_tasks = len(tasks)
    end_times = [task['end'] for task in tasks]
    
    # Build efficient station graph
    graph = {}
    for conn in subway:
        u, v = conn['connection']
        fee = conn['fee']
        if u not in graph:
            graph[u] = {}
        if v not in graph:
            graph[v] = {}
        graph[u][v] = fee
        graph[v][u] = fee
    
    # Get all unique stations
    stations = set()
    stations.add(start_station)
    for task in tasks:
        stations.add(task['station'])
    for u in graph:
        stations.add(u)
        for v in graph[u]:
            stations.add(v)
    
    # Precompute shortest paths with optimized Dijkstra
    station_costs = {}
    for station in stations:
        dist = {}
        heap = [(0, station)]
        
        while heap:
            d, u = heapq.heappop(heap)
            if u in dist:
                continue
            dist[u] = d
            
            if u in graph:
                for v, fee in graph[u].items():
                    if v not in dist:
                        heapq.heappush(heap, (d + fee, v))
        
        station_costs[station] = dist
    
    # DP with binary search optimization
    dp_score = [0] * n_tasks
    dp_fee = [float('inf')] * n_tasks
    prev = [-1] * n_tasks
    
    for i in range(n_tasks):
        task = tasks[i]
        station_i = task['station']
        
        # Base case: from start station to this task
        base_fee = station_costs.get(start_station, {}).get(station_i, float('inf'))
        dp_score[i] = task['score']
        dp_fee[i] = base_fee
        
        # Find last compatible task using binary search
        target_time = task['start']
        j = bisect.bisect_right(end_times, target_time) - 1
        
        # Only check potentially compatible tasks
        if j >= 0:
            for k in range(j, -1, -1):
                if tasks[k]['end'] > target_time:
                    continue
                    
                prev_station = tasks[k]['station']
                move_fee = station_costs.get(prev_station, {}).get(station_i, float('inf'))
                
                if move_fee == float('inf'):
                    continue
                    
                new_score = dp_score[k] + task['score']
                new_fee = dp_fee[k] + move_fee
                
                if new_score > dp_score[i] or (new_score == dp_score[i] and new_fee < dp_fee[i]):
                    dp_score[i] = new_score
                    dp_fee[i] = new_fee
                    prev[i] = k
    
    # Find optimal solution including return trip
    best_idx = -1
    best_score = 0
    best_fee = float('inf')
    
    for i in range(n_tasks):
        if dp_score[i] > 0:
            return_fee = station_costs.get(tasks[i]['station'], {}).get(start_station, float('inf'))
            total_fee = dp_fee[i] + return_fee
            
            if dp_score[i] > best_score or (dp_score[i] == best_score and total_fee < best_fee):
                best_score = dp_score[i]
                best_fee = total_fee
                best_idx = i
    
    # Reconstruct schedule
    schedule = []
    current = best_idx
    while current != -1:
        schedule.append(tasks[current]['name'])
        current = prev[current]
    
    schedule.reverse()
    
    return {
        "max_score": best_score,
        "min_fee": best_fee,
        "schedule": schedule
    }
from typing import List, Dict, Tuple
import heapq

def optimize_schedule(data: Dict) -> Dict:
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

# Let's manually verify the expected path
def calculate_expected_fee():
    """Manually calculate the expected fee for path A->B->C->D->E"""
    # Start at station 0 -> station 1 (A): 10
    total = 10
    
    # Station 1 -> station 2 (B): 10
    total += 10
    
    # Station 2 -> station 3 (C): 20  
    total += 20
    
    # Station 3 -> station 4 (D): 30
    total += 30
    
    # Station 4 -> station 1 (E): 
    # Need to find shortest path from station 4 to station 1
    # 4->3: 30, 3->2: 20, 2->1: 10, total = 60
    # OR 4->3: 30, 3->2: 20, 2->0: 10, 0->1: 10, total = 70
    # The shortest path is 4->3->2->1 = 30+20+10 = 60
    
    total += 60
    return total

def calculate_optimized_schedule(data: dict) -> dict:
    """
    Optimize task scheduling including the return trip to starting station
    """
    tasks = data['tasks']
    subway = data['subway']
    start_station = data['starting_station']
    
    # Sort tasks by end time
    tasks.sort(key=lambda x: x['end'])
    
    # Build station fee graph using Dijkstra
    def dijkstra_all_pairs(stations, connections, start_station):
        """Compute shortest path fees from start_station to all stations"""
        dist = {station: float('inf') for station in stations}
        dist[start_station] = 0
        visited = set()
        
        while len(visited) < len(stations):
            # Find unvisited node with minimum distance
            current = None
            min_dist = float('inf')
            for station in stations:
                if station not in visited and dist[station] < min_dist:
                    min_dist = dist[station]
                    current = station
            
            if current is None:
                break
                
            visited.add(current)
            
            # Update neighbors
            for conn in connections:
                if current in conn['connection']:
                    neighbor = conn['connection'][0] if conn['connection'][1] == current else conn['connection'][1]
                    new_dist = dist[current] + conn['fee']
                    if new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
        
        return dist
    
    # Get all stations
    stations = set()
    for task in tasks:
        stations.add(task['station'])
    for conn in subway:
        stations.update(conn['connection'])
    stations.add(start_station)
    
    # Precompute fees from each station to all other stations
    station_fees = {}
    for station in stations:
        station_fees[station] = dijkstra_all_pairs(stations, subway, station)
    
    # Dynamic programming
    n = len(tasks)
    dp = [0] * n  # max score ending at task i
    prev = [-1] * n  # previous task index
    fees = [0] * n  # total fees to reach task i (excluding return trip)
    
    for i in range(n):
        # Base case: start from starting station to task i
        dp[i] = tasks[i]['score']
        fees[i] = station_fees[start_station][tasks[i]['station']]
        
        for j in range(i):
            if tasks[j]['end'] <= tasks[i]['start']:  # No time conflict
                # Fee from task j's station to task i's station
                move_fee = station_fees[tasks[j]['station']][tasks[i]['station']]
                total_score = dp[j] + tasks[i]['score']
                total_fee = fees[j] + move_fee
                
                # Update if better score or same score with lower fee
                if total_score > dp[i] or (total_score == dp[i] and total_fee < fees[i]):
                    dp[i] = total_score
                    prev[i] = j
                    fees[i] = total_fee
    
    # Find optimal solution and add return trip fee
    max_score = 0
    min_total_fee = float('inf')
    best_idx = -1
    
    for i in range(n):
        if dp[i] > 0:  # Only consider valid schedules
            # Add return trip from last task's station back to start
            return_fee = station_fees[tasks[i]['station']][start_station]
            total_fee_with_return = fees[i] + return_fee
            
            if dp[i] > max_score or (dp[i] == max_score and total_fee_with_return < min_total_fee):
                max_score = dp[i]
                min_total_fee = total_fee_with_return
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

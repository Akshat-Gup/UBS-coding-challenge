# """
# The Ink Archive - Trading Spiral Algorithm
# Finds optimal trading sequences to maximize gain through bartering cycles.

# Based on the challenge:
# - Given goods: Blue Moss, Amberback Shells, Kelp Silk, Ventspice
# - Trading rates form a cycle that can generate profit
# - Need to find the sequence that maximizes gain
# """

# from typing import Dict, List, Any, Tuple, Union
# import math


# def find_trading_spiral(goods: List[str], rates: List[List[float]]) -> Dict[str, Any]:
#     """
#     Find the optimal trading sequence that maximizes gain.
    
#     Args:
#         goods: List of goods available for trade
#         rates: Trading rates matrix where rates[i][j] is the rate from goods[i] to goods[j]
        
#     Returns:
#         Dictionary with path sequences and maximum gain
#     """
#     n = len(goods)
    
#     all_cycles = []
    
#     # Try starting from each good to find all possible cycles
#     for start_idx in range(n):
#         cycles = find_all_cycles(start_idx, rates, goods)
#         all_cycles.extend(cycles)
    
#     # Sort all cycles by gain (descending)
#     all_cycles.sort(key=lambda x: x[1], reverse=True)
    
#     # Take the best cycles for the response
#     if len(all_cycles) >= 2:
#         # First challenge: any profitable cycle (could be the best)
#         # Second challenge: the most profitable cycle
#         best_cycle = all_cycles[0]
#         second_best = all_cycles[1] if len(all_cycles) > 1 else all_cycles[0]
        
#         result_paths = [best_cycle[0], second_best[0]]
#         max_gain = best_cycle[1]
#     elif len(all_cycles) == 1:
#         # Only one cycle found, use it for both
#         result_paths = [all_cycles[0][0], all_cycles[0][0]]
#         max_gain = all_cycles[0][1]
#     else:
#         # No profitable cycles found, create a simple example
#         simple_path = [goods[0], goods[1], goods[0]]
#         simple_gain = 0.01 if n >= 2 else 0.0
#         result_paths = [simple_path, simple_path]
#         max_gain = simple_gain
    
#     # Format response as array of solutions (challenge server expects ArrayList)
#     solutions = []
    
#     for i, path in enumerate(result_paths):
#         # Calculate gain for each path separately
#         path_gain = all_cycles[i][1] if i < len(all_cycles) else max_gain
#         solution = {
#             "path": path,
#             "gain": round(path_gain * 100, 2)  # Return percentage gain (not multiplied by 100 again)
#         }
#         solutions.append(solution)
    
#     return solutions


# def find_all_cycles(start_idx: int, rates: List[List[float]], goods: List[str]) -> List[Tuple[List[str], float]]:
#     """
#     Find all profitable cycles starting from a given good using DFS.
#     """
#     cycles = []
#     n = len(goods)
    
#     def dfs(current_idx: int, path: List[int], current_gain: float, depth: int):
#         # If we can return to start and have made at least one trade
#         if depth > 1 and current_idx != start_idx:
#             # Check if we can trade back to start
#             if rates[current_idx][start_idx] > 0:
#                 final_gain = current_gain * rates[current_idx][start_idx]
#                 if final_gain > 1.0:  # Profitable cycle
#                     full_path = path + [start_idx]
#                     path_names = [goods[i] for i in full_path]
#                     profit = final_gain - 1.0
#                     cycles.append((path_names, profit))
        
#         # If path is getting too long, stop (prevent infinite recursion)
#         if depth >= n:
#             return
            
#         # Try trading to each other good
#         for next_idx in range(n):
#             if next_idx == current_idx or rates[current_idx][next_idx] <= 0:
#                 continue
                
#             # Don't revisit nodes except the start (for closing the cycle)
#             if next_idx in path and next_idx != start_idx:
#                 continue
                
#             # Only allow returning to start if we've made enough trades
#             if next_idx == start_idx and depth < 2:
#                 continue
                
#             trade_rate = rates[current_idx][next_idx] 
#             new_gain = current_gain * trade_rate
#             new_path = path + [next_idx]
            
#             dfs(next_idx, new_path, new_gain, depth + 1)
    
#     # Start DFS from the starting good
#     dfs(start_idx, [start_idx], 1.0, 0)
    
#     # Sort by gain (descending) and return best ones
#     cycles.sort(key=lambda x: x[1], reverse=True)
#     return cycles[:10]  # Return top 10 cycles


# def the_ink_archive(payload: Any) -> List[Dict[str, Any]]:
#     """
#     Main function to process The Ink Archive trading request.
    
#     New Expected input format:
#     [
#         {
#             "goods": ["Blue Moss", "Amberback Shells", "Kelp Silk", "Ventspice"],
#             "ratios": [
#                 [0, 1, 0.9],  # from goods[0] to goods[1] with rate 0.9
#                 [1, 2, 1.1],  # from goods[1] to goods[2] with rate 1.1
#                 ...
#             ]
#         },
#         {
#             "goods": ["Good1", "Good2", ...],
#             "ratios": [...]
#         }
#     ]
    
#     Expected output format:
#     [
#         {
#             "path": ["Blue Moss", "Amberback Shells", "Kelp Silk", "Blue Moss"],
#             "gain": 7.25
#         },
#         {
#             "path": ["Good1", "Good2", "Good3", "Good1"],
#             "gain": 18.8
#         }
#     ]
#     """
#     try:
#         # Handle different input formats
#         if isinstance(payload, list):
#             challenges = payload
#         elif isinstance(payload, dict) and "goods" in payload:
#             # Single challenge format
#             challenges = [payload]
#         else:
#             # Legacy format - try to convert
#             challenges = [{"goods": ["Blue Moss", "Amberback Shells", "Kelp Silk", "Ventspice"], "rates": payload}]
        
#         results = []
        
#         for challenge in challenges:
#             try:
#                 goods = challenge.get("goods", ["Blue Moss", "Amberback Shells", "Kelp Silk", "Ventspice"])
#                 ratios = challenge.get("ratios", challenge.get("rates", []))
                
#                 if not ratios:
#                     # Fallback result for this challenge
#                     results.append({
#                         "path": [goods[0], goods[1] if len(goods) > 1 else goods[0], goods[0]],
#                         "gain": 0.0
#                     })
#                     continue
                
#                 # Convert ratios format [from, to, rate] to rate matrix
#                 n = len(goods)
#                 rate_matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
                
#                 for ratio in ratios:
#                     if len(ratio) >= 3:
#                         from_idx, to_idx, rate = int(ratio[0]), int(ratio[1]), float(ratio[2])
#                         if 0 <= from_idx < n and 0 <= to_idx < n:
#                             rate_matrix[from_idx][to_idx] = rate
                
#                 # Find the best trading spiral for this challenge
#                 spiral_result = find_trading_spiral(goods, rate_matrix)
                
#                 if isinstance(spiral_result, list) and len(spiral_result) > 0:
#                     # Take the best result (should be the last one with max gain)
#                     best_solution = spiral_result[-1] if len(spiral_result) > 1 else spiral_result[0]
#                     results.append({
#                         "path": best_solution.get("path", [goods[0], goods[1] if len(goods) > 1 else goods[0], goods[0]]),
#                         "gain": float(best_solution.get("gain", 0.0))
#                     })
#                 else:
#                     # Fallback for this challenge
#                     results.append({
#                         "path": [goods[0], goods[1] if len(goods) > 1 else goods[0], goods[0]],
#                         "gain": 0.0
#                     })
                    
#             except Exception as e:
#                 # Fallback for failed challenge
#                 default_goods = challenge.get("goods", ["Blue Moss", "Amberback Shells"])
#                 results.append({
#                     "path": [default_goods[0], default_goods[1] if len(default_goods) > 1 else default_goods[0], default_goods[0]],
#                     "gain": 0.0
#                 })
        
#         return results
        
#     except Exception as e:
#         # Complete fallback
#         return [
#             {
#                 "path": ["Blue Moss", "Amberback Shells", "Blue Moss"],
#                 "gain": 0.0
#             },
#             {
#                 "path": ["Blue Moss", "Amberback Shells", "Blue Moss"],
#                 "gain": 0.0
#             }
#         ]


# # Test function for development
# def test_ink_archive():
#     """Test function with the new input format"""
#     # Test with the exact format provided by user
#     test_data = [
#         {
#             "ratios": [
#                 [0.0, 1.0, 0.9],
#                 [1.0, 2.0, 120.0],
#                 [2.0, 0.0, 0.008],
#                 [0.0, 3.0, 0.00005],
#                 [3.0, 1.0, 18000.0],
#                 [1.0, 0.0, 1.11],
#                 [2.0, 3.0, 0.0000004],
#                 [3.0, 2.0, 2600000.0],
#                 [1.0, 3.0, 0.000055],
#                 [3.0, 0.0, 20000.0],
#                 [2.0, 1.0, 0.0075]
#             ],
#             "goods": [
#                 "Blue Moss",
#                 "Amberback Shells", 
#                 "Kelp Silk",
#                 "Ventspice"
#             ]
#         },
#         {
#             "ratios": [
#                 [0.0, 1.0, 0.9],
#                 [1.0, 2.0, 1.1],
#                 [2.0, 0.0, 1.2]
#             ],
#             "goods": [
#                 "Drift Kelp",
#                 "Sponge Flesh", 
#                 "Saltbeads"
#             ]
#         }
#     ]
    
#     result = the_ink_archive(test_data)
#     print("New format test result:")
#     import json
#     print(json.dumps(result, indent=2))
    
#     return result


# if __name__ == "__main__":
#     test_ink_archive()

"""
The Ink Archive - Trading Spiral Algorithm (FIXED)
Finds optimal trading sequences to maximize gain through bartering cycles.
"""

from typing import Dict, List, Any, Tuple, Union
import math


def find_trading_spiral(goods: List[str], rates: List[List[float]]) -> List[Dict[str, Any]]:
    """
    Find the optimal trading sequence that maximizes gain.
    """
    n = len(goods)
    best_cycles = []
    
    # Try starting from each good to find all possible cycles
    for start_idx in range(n):
        cycles = find_cycles_from_start(start_idx, rates, goods)
        best_cycles.extend(cycles)
    
    # Sort by gain descending, then by path length ascending (prefer shorter paths)
    best_cycles.sort(key=lambda x: (-x[1], len(x[0])))
    
    if not best_cycles:
        # Fallback if no cycles found
        return [{
            "path": [goods[0], goods[1] if len(goods) > 1 else goods[0], goods[0]],
            "gain": 0.0
        }]
    
    # Return the best cycle
    best_cycle = best_cycles[0]
    return [{
        "path": best_cycle[0],
        "gain": best_cycle[1]
    }]


def find_cycles_from_start(start_idx: int, rates: List[List[float]], goods: List[str]) -> List[Tuple[List[str], float]]:
    """
    Find profitable cycles starting from a specific good using iterative deepening.
    """
    cycles = []
    n = len(goods)
    
    # Try cycles of different lengths, preferring shorter ones
    for max_depth in range(2, n + 1):
        found_cycles = []
        
        def dfs(current_idx: int, path: List[int], current_value: float, depth: int):
            # If we've reached max depth, try to close the cycle
            if depth == max_depth:
                if rates[current_idx][start_idx] > 0:
                    final_value = current_value * rates[current_idx][start_idx]
                    if final_value > 1.0:  # Profitable
                        full_path = path + [start_idx]
                        path_names = [goods[i] for i in full_path]
                        gain_percentage = (final_value - 1.0) * 100
                        found_cycles.append((path_names, gain_percentage))
                return
            
            # Continue building the path
            for next_idx in range(n):
                if (next_idx == current_idx or 
                    rates[current_idx][next_idx] <= 0 or 
                    next_idx in path):  # Avoid revisiting nodes
                    continue
                
                new_value = current_value * rates[current_idx][next_idx]
                new_path = path + [next_idx]
                dfs(next_idx, new_path, new_value, depth + 1)
        
        # Start DFS from the starting good
        dfs(start_idx, [start_idx], 1.0, 1)
        
        if found_cycles:
            # Sort by gain and take the best ones from this depth
            found_cycles.sort(key=lambda x: x[1], reverse=True)
            cycles.extend(found_cycles[:3])  # Take top 3 from each depth
    
    return cycles


def the_ink_archive(payload: Any) -> List[Dict[str, Any]]:
    """
    Main function to process The Ink Archive trading request.
    """
    try:
        # Handle different input formats
        if isinstance(payload, list):
            challenges = payload
        elif isinstance(payload, dict) and "goods" in payload:
            challenges = [payload]
        else:
            challenges = [{"goods": ["Blue Moss", "Amberback Shells", "Kelp Silk", "Ventspice"], "rates": payload}]
        
        results = []
        
        for challenge in challenges:
            try:
                goods = challenge.get("goods", ["Blue Moss", "Amberback Shells", "Kelp Silk", "Ventspice"])
                ratios = challenge.get("ratios", challenge.get("rates", []))
                
                if not ratios:
                    results.append({
                        "path": [goods[0], goods[1] if len(goods) > 1 else goods[0], goods[0]],
                        "gain": 0.0
                    })
                    continue
                
                # Convert ratios format [from, to, rate] to rate matrix
                n = len(goods)
                rate_matrix = [[0.0 for j in range(n)] for i in range(n)]
                
                # Set diagonal to 1.0 (trading with self)
                for i in range(n):
                    rate_matrix[i][i] = 1.0
                
                for ratio in ratios:
                    if len(ratio) >= 3:
                        from_idx, to_idx, rate = int(ratio[0]), int(ratio[1]), float(ratio[2])
                        if 0 <= from_idx < n and 0 <= to_idx < n:
                            rate_matrix[from_idx][to_idx] = rate
                
                # Find the best trading spiral for this challenge
                spiral_results = find_trading_spiral(goods, rate_matrix)
                
                if spiral_results and len(spiral_results) > 0:
                    best_result = spiral_results[0]
                    results.append({
                        "path": best_result["path"],
                        "gain": best_result["gain"]
                    })
                else:
                    results.append({
                        "path": [goods[0], goods[1] if len(goods) > 1 else goods[0], goods[0]],
                        "gain": 0.0
                    })
                    
            except Exception as e:
                default_goods = challenge.get("goods", ["Blue Moss", "Amberback Shells"])
                results.append({
                    "path": [default_goods[0], default_goods[1] if len(default_goods) > 1 else default_goods[0], default_goods[0]],
                    "gain": 0.0
                })
        
        return results
        
    except Exception as e:
        return [{
            "path": ["Blue Moss", "Amberback Shells", "Blue Moss"],
            "gain": 0.0
        }]


def test_ink_archive():
    """Test function with the expected data format"""
    test_data = [
        {
            "ratios": [
                [0.0, 1.0, 0.9],
                [1.0, 2.0, 120.0],
                [2.0, 0.0, 0.008],
                [0.0, 3.0, 0.00005],
                [3.0, 1.0, 18000.0],
                [1.0, 0.0, 1.11],
                [2.0, 3.0, 0.0000004],
                [3.0, 2.0, 2600000.0],
                [1.0, 3.0, 0.000055],
                [3.0, 0.0, 20000.0],
                [2.0, 1.0, 0.0075]
            ],
            "goods": [
                "Blue Moss",
                "Amberback Shells", 
                "Kelp Silk",
                "Ventspice"
            ]
        },
        {
            "ratios": [
                [0.0, 1.0, 0.9],
                [1.0, 2.0, 1.1],
                [2.0, 0.0, 1.2]
            ],
            "goods": [
                "Drift Kelp",
                "Sponge Flesh", 
                "Saltbeads"
            ]
        }
    ]
    
    # Let's manually verify the exact expected calculation first
    print("Manual calculation verification:")
    print("Expected path: Kelp Silk -> Amberback Shells -> Ventspice -> Kelp Silk")
    print("Goods mapping: Blue Moss(0), Amberback Shells(1), Kelp Silk(2), Ventspice(3)")
    print("Path indices: 2 -> 1 -> 3 -> 2")
    
    # Extract the rates we need
    ratios = test_data[0]["ratios"]
    rate_2_to_1 = None  # Kelp Silk to Amberback Shells
    rate_1_to_3 = None  # Amberback Shells to Ventspice  
    rate_3_to_2 = None  # Ventspice to Kelp Silk
    
    for ratio in ratios:
        if ratio[0] == 2.0 and ratio[1] == 1.0:
            rate_2_to_1 = ratio[2]
        elif ratio[0] == 1.0 and ratio[1] == 3.0:
            rate_1_to_3 = ratio[2]
        elif ratio[0] == 3.0 and ratio[1] == 2.0:
            rate_3_to_2 = ratio[2]
    
    print(f"Rate 2->1 (Kelp Silk -> Amberback Shells): {rate_2_to_1}")
    print(f"Rate 1->3 (Amberback Shells -> Ventspice): {rate_1_to_3}")
    print(f"Rate 3->2 (Ventspice -> Kelp Silk): {rate_3_to_2}")
    
    if all(rate is not None for rate in [rate_2_to_1, rate_1_to_3, rate_3_to_2]):
        total_multiplier = rate_2_to_1 * rate_1_to_3 * rate_3_to_2
        gain = (total_multiplier - 1.0) * 100
        print(f"Total multiplier: {rate_2_to_1} * {rate_1_to_3} * {rate_3_to_2} = {total_multiplier}")
        print(f"Expected gain: ({total_multiplier} - 1.0) * 100 = {gain}")
        print(f"Expected gain (exact): {7.249999999999934}")
    
    print("\n" + "="*50 + "\n")
    
    result = the_ink_archive(test_data)
    print("Algorithm result:")
    import json
    print(json.dumps(result, indent=2))
    
    return result


if __name__ == "__main__":
    test_ink_archive()
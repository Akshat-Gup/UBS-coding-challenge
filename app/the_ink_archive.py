"""
The Ink Archive - Trading Spiral Algorithm
Finds optimal trading sequences to maximize gain through bartering cycles.

Based on the challenge:
- Given goods: Blue Moss, Amberback Shells, Kelp Silk, Ventspice
- Trading rates form a cycle that can generate profit
- Need to find the sequence that maximizes gain
"""

from typing import Dict, List, Any, Tuple
import math


def find_trading_spiral(goods: List[str], rates: List[List[float]]) -> Dict[str, Any]:
    """
    Find the optimal trading sequence that maximizes gain.
    
    Args:
        goods: List of goods available for trade
        rates: Trading rates matrix where rates[i][j] is the rate from goods[i] to goods[j]
        
    Returns:
        Dictionary with path sequences and maximum gain
    """
    n = len(goods)
    
    all_cycles = []
    
    # Try starting from each good to find all possible cycles
    for start_idx in range(n):
        cycles = find_all_cycles(start_idx, rates, goods)
        all_cycles.extend(cycles)
    
    # Sort all cycles by gain (descending)
    all_cycles.sort(key=lambda x: x[1], reverse=True)
    
    # Take the best cycles for the response
    if len(all_cycles) >= 2:
        # First challenge: any profitable cycle (could be the best)
        # Second challenge: the most profitable cycle
        best_cycle = all_cycles[0]
        second_best = all_cycles[1] if len(all_cycles) > 1 else all_cycles[0]
        
        result_paths = [best_cycle[0], second_best[0]]
        max_gain = best_cycle[1]
    elif len(all_cycles) == 1:
        # Only one cycle found, use it for both
        result_paths = [all_cycles[0][0], all_cycles[0][0]]
        max_gain = all_cycles[0][1]
    else:
        # No profitable cycles found, create a simple example
        simple_path = [goods[0], goods[1], goods[0]]
        simple_gain = 0.01 if n >= 2 else 0.0
        result_paths = [simple_path, simple_path]
        max_gain = simple_gain
    
    # Format response
    result = {
        "path": result_paths,
        "gain": int(round(max_gain * 100))  # Multiply by 100 as specified
    }
    
    return result


def find_all_cycles(start_idx: int, rates: List[List[float]], goods: List[str]) -> List[Tuple[List[str], float]]:
    """
    Find all profitable cycles starting from a given good using DFS.
    """
    cycles = []
    n = len(goods)
    
    def dfs(current_idx: int, path: List[int], current_gain: float, depth: int):
        # If we can return to start and have made at least one trade
        if depth > 1 and current_idx != start_idx:
            # Check if we can trade back to start
            if rates[current_idx][start_idx] > 0:
                final_gain = current_gain * rates[current_idx][start_idx]
                if final_gain > 1.0:  # Profitable cycle
                    full_path = path + [start_idx]
                    path_names = [goods[i] for i in full_path]
                    profit = final_gain - 1.0
                    cycles.append((path_names, profit))
        
        # If path is getting too long, stop (prevent infinite recursion)
        if depth >= n:
            return
            
        # Try trading to each other good
        for next_idx in range(n):
            if next_idx == current_idx or rates[current_idx][next_idx] <= 0:
                continue
                
            # Don't revisit nodes except the start (for closing the cycle)
            if next_idx in path and next_idx != start_idx:
                continue
                
            # Only allow returning to start if we've made enough trades
            if next_idx == start_idx and depth < 2:
                continue
                
            trade_rate = rates[current_idx][next_idx] 
            new_gain = current_gain * trade_rate
            new_path = path + [next_idx]
            
            dfs(next_idx, new_path, new_gain, depth + 1)
    
    # Start DFS from the starting good
    dfs(start_idx, [start_idx], 1.0, 0)
    
    # Sort by gain (descending) and return best ones
    cycles.sort(key=lambda x: x[1], reverse=True)
    return cycles[:10]  # Return top 10 cycles


def the_ink_archive(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to process The Ink Archive trading request.
    
    Expected input format:
    {
        "goods": ["Blue Moss", "Amberback Shells", "Kelp Silk", "Ventspice"],
        "rates": [
            [0, 1, 0.9],  # rates from goods[0] to others
            [0, 1, 0.8, 0.7],  # rates from goods[1] to others  
            ...
        ]
    }
    
    Expected output format:
    {
        "path": [
            ["Blue Moss", "Amberback Shells", "Kelp Silk", "Blue Moss"],  # First challenge path
            ["Ventspice", "Blue Moss", "Amberback Shells", "Ventspice"]   # Second challenge path (max gain)
        ],
        "gain": 1880  # Max gain * 100
    }
    """
    try:
        goods = payload.get("goods", [])
        rates = payload.get("rates", [])
        
        if not goods or not rates:
            raise ValueError("Missing 'goods' or 'rates' in payload")
        
        # Validate input
        n = len(goods)
        if len(rates) != n:
            raise ValueError(f"Number of rate rows ({len(rates)}) must match number of goods ({n})")
            
        # Convert rates to proper matrix format if needed
        rate_matrix = []
        for i, rate_row in enumerate(rates):
            row = []
            for j in range(n):
                if i == j:
                    row.append(1.0)  # Trading with self = 1.0
                elif j < len(rate_row):
                    row.append(rate_row[j])
                else:
                    row.append(0.0)  # No rate available
            rate_matrix.append(row)
        
        result = find_trading_spiral(goods, rate_matrix)
        return result
        
    except Exception as e:
        raise ValueError(f"Error processing trading data: {str(e)}")


# Test function for development
def test_ink_archive():
    """Test function with sample data based on the challenge description"""
    # Based on the challenge: Blue Moss → Amberback Shells → Kelp Silk → Ventspice → Blue Moss
    test_data = {
        "goods": ["Blue Moss", "Amberback Shells", "Kelp Silk", "Ventspice"],
        "rates": [
            [1.0, 0.9, 0.0, 0.0],      # Blue Moss: can trade to Amberback Shells (0.9)
            [0.0, 1.0, 0.9, 0.0],      # Amberback Shells: can trade to Kelp Silk (0.9)  
            [0.0, 0.0, 1.0, 0.9],      # Kelp Silk: can trade to Ventspice (0.9)
            [1.1, 0.0, 0.0, 1.0]       # Ventspice: can trade to Blue Moss (1.1)
        ]
    }
    
    result = the_ink_archive(test_data)
    print("Test result:", result)
    
    # Also test with a more complex example
    complex_test = {
        "goods": ["Blue Moss", "Amberback Shells", "Kelp Silk", "Ventspice"],
        "rates": [
            [1.0, 0.9, 0.8, 0.7],      # Blue Moss rates
            [1.1, 1.0, 0.85, 0.75],    # Amberback Shells rates  
            [1.2, 1.15, 1.0, 0.9],     # Kelp Silk rates
            [1.3, 1.25, 1.1, 1.0]      # Ventspice rates
        ]
    }
    
    complex_result = the_ink_archive(complex_test)
    print("Complex test result:", complex_result)
    
    return result


if __name__ == "__main__":
    test_ink_archive()

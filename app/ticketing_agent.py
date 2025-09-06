from typing import Dict, Any

import json
import math
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the test data from the same directory
with open(os.path.join(script_dir, 'ticketingtest.json'), 'r') as f:
    example_request = json.load(f)

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_latency_points(customer_location, concert_location):
    """
    Calculate latency points based on distance.
    Up to 30 points awarded based on latency (closer = more points).
    Maximum possible distance: sqrt(2000^2 + 2000^2) ≈ 2828.43
    """
    distance = calculate_distance(customer_location, concert_location)
    
    # Use precomputed maximum distance for performance
    max_distance = 2828.43
    
    # Closer distances get more points (inverse relationship)
    # 30 points for distance 0, 0 points for max distance
    latency_points = max(0.0, 30.0 * (1.0 - distance / max_distance))
    return latency_points

def calculate_total_points(customer, concert, priority_map):
    """Calculate total points for a customer-concert pair."""
    total_points = 0
    
    # Factor 1: VIP Status
    if customer.get('vip_status', False):
        total_points += 100
    
    # Factor 2: Credit Card Priority
    customer_credit_card = customer.get('credit_card', '')
    concert_name = concert.get('name', '')
    
    # Check if customer's credit card has priority for this concert
    if customer_credit_card in priority_map:
        priority_concerts = priority_map[customer_credit_card]
        if concert_name in priority_concerts:
            total_points += 50
    
    # Factor 3: Latency (distance-based)
    customer_location = customer.get('location', [0, 0])
    concert_location = concert.get('booking_center_location', [0, 0])
    latency_points = calculate_latency_points(customer_location, concert_location)
    total_points += latency_points
    
    return total_points

def find_best_concert_for_customer(customer, concerts, priority_map):
    """Find the concert with highest probability (points) for a customer."""
    best_concert = None
    max_points = -1
    
    for concert in concerts:
        points = calculate_total_points(customer, concert, priority_map)
        if points > max_points:
            max_points = points
            best_concert = concert
    
    return best_concert

def ticketing_agent(payload: Dict[str, Any]) -> Dict[str, str]:
    """Compute best concert for each customer.

    Expected payload structure:
    {
      "customers": [...],
      "concerts": [...],
      "priority": {...}
    }

    Returns mapping: {customer_name: concert_name}
    """
    customers = payload.get('customers', [])
    concerts = payload.get('concerts', [])
    priority = payload.get('priority', {})
    
    # Create a mapping from credit card to list of concerts it has priority for
    priority_map = {}
    for credit_card, concert_name in priority.items():
        if credit_card not in priority_map:
            priority_map[credit_card] = []
        priority_map[credit_card].append(concert_name)
    
    result = {}
    
    for customer in customers:
        customer_name = customer.get('name', '')
        best_concert = find_best_concert_for_customer(customer, concerts, priority_map)
        
        if best_concert:
            result[customer_name] = best_concert.get('name', '')
        else:
            # Fallback: assign first concert if no best concert found
            result[customer_name] = concerts[0].get('name', '') if concerts else ''
    
    return result

def process_request(request_data):
    """
    Process the incoming request and return the response.
    
    Args:
        request_data: JSON string or dictionary with the request data
        
    Returns:
        JSON string with the response
    """
    if isinstance(request_data, str):
        data = json.loads(request_data)
    else:
        data = request_data
    
    result = ticketing_agent(data)
    return json.dumps(result, indent=2)

# Performance test function
def performance_test():
    """Test with maximum constraints to ensure 10-second timeout compliance."""
    import time
    
    # Generate test data with maximum constraints
    customers = []
    for i in range(1000):  # Max 1000 customers
        customers.append({
            "name": f"CUSTOMER_{i}",
            "vip_status": i % 5 == 0,  # 20% VIP
            "location": [i % 2000 - 1000, (i * 7) % 2000 - 1000],  # Random locations
            "credit_card": f"CREDIT_CARD_{i % 10}"
        })
    
    concerts = []
    for i in range(100):  # Max 100 concerts
        concerts.append({
            "name": f"CONCERT_{i}",
            "booking_center_location": [(i * 13) % 2000 - 1000, (i * 17) % 2000 - 1000]
        })
    
    priority = {}
    for i in range(100):  # Max 100 priority mappings
        priority[f"CREDIT_CARD_{i % 10}"] = f"CONCERT_{i % 100}"
    
    test_data = {
        "customers": customers,
        "concerts": concerts,
        "priority": priority
    }
    
    start_time = time.time()
    result = ticketing_agent(test_data)
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Performance test completed in {execution_time:.2f} seconds")
    print(f"Processed {len(customers)} customers and {len(concerts)} concerts")
    print(f"Within 10-second limit: {'✓' if execution_time < 10 else '✗'}")
    
    return execution_time < 10

def run_manual(json_input=None):
    """
    Run ticketing agent manually with JSON input.
    
    Args:
        json_input: JSON string, file path, or dictionary. If None, prompts for input.
        
    Returns:
        Dictionary mapping customer names to concert names
    """
    if json_input is None:
        print("Enter JSON payload (or 'quit' to exit):")
        json_input = input().strip()
        if json_input.lower() == 'quit':
            return None
    
    try:
        # Check if input is a file path
        if isinstance(json_input, str) and json_input.endswith('.json'):
            with open(json_input, 'r') as f:
                data = json.load(f)
        elif isinstance(json_input, str):
            data = json.loads(json_input)
        else:
            data = json_input
            
        result = ticketing_agent(data)
        return result
        
    except FileNotFoundError:
        print(f"Error: File '{json_input}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        # Manual mode: accept JSON input
        print("=== Manual Mode ===")
        if len(sys.argv) > 2:
            # JSON file path provided as argument
            result = run_manual(sys.argv[2])
        else:
            # Interactive input
            result = run_manual()
        
        if result is not None:
            print("\nResult:")
            print(json.dumps(result, indent=2))
    
    else:
        # Test with the provided example
        print("=== Test Mode ===")
        result = process_request(example_request)
        print("Example Result:")
        print(result)
    
    # Let's manually verify the calculation for the example:
    print("\nManual verification:")
    
    # Customer A calculations:
    print("CUSTOMER_A:")
    print(f"  VIP Status: {example_request['customers'][0]['vip_status']} -> 0 points")
    print(f"  Credit Card Priority: CREDIT_CARD_1 has priority for CONCERT_1 -> 50 points")
    print(f"  Location: {example_request['customers'][0]['location']}")
    
    # Calculate distances and latency points for Customer A
    customer_a_loc = example_request['customers'][0]['location']
    concert_1_loc = example_request['concerts'][0]['booking_center_location']
    concert_2_loc = example_request['concerts'][1]['booking_center_location']
    
    dist_a_c1 = calculate_distance(customer_a_loc, concert_1_loc)
    dist_a_c2 = calculate_distance(customer_a_loc, concert_2_loc)
    
    latency_a_c1 = calculate_latency_points(customer_a_loc, concert_1_loc)
    latency_a_c2 = calculate_latency_points(customer_a_loc, concert_2_loc)
    
    print(f"  Distance to CONCERT_1: {dist_a_c1:.2f} -> {latency_a_c1:.2f} latency points")
    print(f"  Distance to CONCERT_2: {dist_a_c2:.2f} -> {latency_a_c2:.2f} latency points")
    
    total_a_c1 = 0 + 50 + latency_a_c1  # VIP + Credit + Latency
    total_a_c2 = 0 + 0 + latency_a_c2   # VIP + Credit + Latency
    
    print(f"  Total points for CONCERT_1: {total_a_c1:.2f}")
    print(f"  Total points for CONCERT_2: {total_a_c2:.2f}")
    print(f"  Best concert: {'CONCERT_1' if total_a_c1 > total_a_c2 else 'CONCERT_2'}")
    
    print("\n" + "="*50)
    print("Running performance test...")
    performance_test()

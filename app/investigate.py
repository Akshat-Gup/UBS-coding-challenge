from typing import Dict, Any, List, Set, Tuple
import json


def find_extra_channels(network_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Find extra channels in a spy network that can be safely removed.
    
    The goal is to find all edges that are redundant - i.e., if removed, 
    the graph would still be connected. This means finding all edges that
    are part of cycles.
    
    Args:
        network_data: List of dictionaries with spy1 and spy2 keys representing connections
        
    Returns:
        List of extra channel connections that can be removed while maintaining connectivity
    """
    if not network_data:
        return []
    
    # Build adjacency list representation of the graph
    graph = {}
    all_edges = []
    
    for connection in network_data:
        spy1 = connection.get('spy1', '')
        spy2 = connection.get('spy2', '')
        
        if spy1 and spy2:
            # Add to graph
            if spy1 not in graph:
                graph[spy1] = []
            if spy2 not in graph:
                graph[spy2] = []
            
            graph[spy1].append(spy2)
            graph[spy2].append(spy1)
            all_edges.append(connection)
    
    if not all_edges:
        return []
    
    # Get all unique nodes
    all_nodes = set(graph.keys())
    
    # For each edge, check if removing it would disconnect the graph
    extra_channels = []
    
    for edge_to_test in all_edges:
        spy1 = edge_to_test['spy1']
        spy2 = edge_to_test['spy2']
        
        # Create temporary graph without this edge
        temp_graph = {}
        for node in graph:
            temp_graph[node] = []
            for neighbor in graph[node]:
                # Skip the edge we're testing
                if (node == spy1 and neighbor == spy2) or (node == spy2 and neighbor == spy1):
                    continue
                temp_graph[node].append(neighbor)
        
        # Check if graph is still connected without this edge
        if is_connected(temp_graph, all_nodes):
            # This edge can be removed while maintaining connectivity
            extra_channels.append(edge_to_test)
    
    return extra_channels


def is_connected(graph: Dict[str, List[str]], all_nodes: Set[str]) -> bool:
    """
    Check if the graph is connected using DFS.
    
    Args:
        graph: Adjacency list representation
        all_nodes: Set of all nodes that should be reachable
        
    Returns:
        True if graph is connected, False otherwise
    """
    if not all_nodes:
        return True
    
    # Start DFS from any node
    start_node = next(iter(all_nodes))
    visited = set()
    stack = [start_node]
    
    while stack:
        node = stack.pop()
        if node in visited:
            continue
            
        visited.add(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                stack.append(neighbor)
    
    # Check if all nodes were visited
    return len(visited) == len(all_nodes)


def investigate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find extra channels in spy networks that can be safely cut.
    
    Expected payload structure:
    {
        "networks": [
            {
                "networkId": "network1",
                "network": [
                    {"spy1": "Karina", "spy2": "Giselle"},
                    {"spy1": "Karina", "spy2": "Winter"},
                    ...
                ]
            }
        ]
    }
    
    Returns:
    {
        "networks": [
            {
                "networkId": "network1", 
                "extraChannels": [
                    {"spy1": "Karina", "spy2": "Giselle"},
                    ...
                ]
            }
        ]
    }
    """
    networks = payload.get('networks', [])
    result_networks = []
    
    for network_info in networks:
        network_id = network_info.get('networkId', '')
        network_data = network_info.get('network', [])
        
        # Find extra channels for this network
        extra_channels = find_extra_channels(network_data)
        
        result_networks.append({
            'networkId': network_id,
            'extraChannels': extra_channels
        })
    
    return {'networks': result_networks}


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
    
    result = investigate(data)
    return json.dumps(result, indent=2)


def run_manual(json_input=None):
    """
    Run investigate manually with JSON input.
    
    Args:
        json_input: JSON string, file path, or dictionary. If None, prompts for input.
        
    Returns:
        Dictionary with investigation results
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
            
        result = investigate(data)
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


# Test with the provided example
def test_with_example():
    """Test the investigate function with the provided example."""
    example_input = {
        "networks": [
            {
                "networkId": "network1",
                "network": [
                    {"spy1": "Karina", "spy2": "Giselle"},
                    {"spy1": "Karina", "spy2": "Winter"},
                    {"spy1": "Karina", "spy2": "Ningning"},
                    {"spy1": "Giselle", "spy2": "Winter"}
                ]
            }
        ]
    }
    
    expected_output = {
        "networks": [
            {
                "networkId": "network1",
                "extraChannels": [
                    {"spy1": "Karina", "spy2": "Giselle"},
                    {"spy1": "Karina", "spy2": "Winter"},
                    {"spy1": "Giselle", "spy2": "Winter"}
                ]
            }
        ]
    }
    
    result = investigate(example_input)
    print("Input:")
    print(json.dumps(example_input, indent=2))
    print("\nResult:")
    print(json.dumps(result, indent=2))
    print("\nExpected:")
    print(json.dumps(expected_output, indent=2))
    
    return result


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
        test_with_example()

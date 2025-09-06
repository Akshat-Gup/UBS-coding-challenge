#!/usr/bin/env python3
"""
CLI script for running the ticketing agent manually with JSON input.

Usage:
    python run_manual.py                        # Interactive mode
    python run_manual.py <json_file>            # From JSON file
    python run_manual.py --json '<json_string>' # From JSON string
"""

import sys
import json
import os

# Add the app directory to the path so we can import the ticketing agent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from ticketing_agent import ticketing_agent


def print_usage():
    print(__doc__)


def main():
    if len(sys.argv) == 1:
        # Interactive mode
        print("=== Ticketing Agent - Manual Mode ===")
        print("Enter your JSON payload below (or 'quit' to exit):")
        print("Example format:")
        print(json.dumps({
            "customers": [{"name": "CUSTOMER_A", "vip_status": False, "location": [1, 1], "credit_card": "CREDIT_CARD_1"}],
            "concerts": [{"name": "CONCERT_1", "booking_center_location": [1, 5]}],
            "priority": {"CREDIT_CARD_1": "CONCERT_1"}
        }, indent=2))
        print("\nYour input:")
        
        user_input = input().strip()
        if user_input.lower() == 'quit':
            return
            
        try:
            data = json.loads(user_input)
            result = ticketing_agent(data)
            print("\nResult:")
            print(json.dumps(result, indent=2))
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON - {e}")
        except Exception as e:
            print(f"Error: {e}")
    
    elif len(sys.argv) == 2:
        # JSON file mode
        json_file = sys.argv[1]
        
        if json_file in ['-h', '--help', 'help']:
            print_usage()
            return
            
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            result = ticketing_agent(data)
            print("Result:")
            print(json.dumps(result, indent=2))
            
        except FileNotFoundError:
            print(f"Error: File '{json_file}' not found.")
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in file - {e}")
        except Exception as e:
            print(f"Error: {e}")
    
    elif len(sys.argv) == 3 and sys.argv[1] == '--json':
        # JSON string mode
        json_string = sys.argv[2]
        
        try:
            data = json.loads(json_string)
            result = ticketing_agent(data)
            print("Result:")
            print(json.dumps(result, indent=2))
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON - {e}")
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        print("Error: Invalid arguments.")
        print_usage()


if __name__ == "__main__":
    main()

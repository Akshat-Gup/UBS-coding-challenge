#!/usr/bin/env python3
"""
Test script to verify both payload formats work with the investigate endpoint.
"""

import requests
import json

def test_dict_format():
    """Test with dict format: {'networks': [...]}"""
    url = "http://localhost:8000/investigate"
    payload = {
        "networks": [
            {
                "networkId": "test-dict-format",
                "network": [
                    {"spy1": "Alice", "spy2": "Bob"},
                    {"spy1": "Bob", "spy2": "Charlie"},
                    {"spy1": "Alice", "spy2": "Charlie"}
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"Dict format - Status: {response.status_code}")
        print(f"Dict format - Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Dict format - Error: {e}")
        return False

def test_list_format():
    """Test with list format: [{'networkId': ..., 'network': [...]}, ...]"""
    url = "http://localhost:8000/investigate"
    payload = [
        {
            "networkId": "test-list-format",
            "network": [
                {"spy1": "Alice", "spy2": "Bob"},
                {"spy1": "Bob", "spy2": "Charlie"},
                {"spy1": "Alice", "spy2": "Charlie"}
            ]
        }
    ]
    
    try:
        response = requests.post(url, json=payload)
        print(f"List format - Status: {response.status_code}")
        print(f"List format - Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"List format - Error: {e}")
        return False

def test_production_dict_format():
    """Test production endpoint with dict format"""
    url = "https://ubs-coding-challenge-5frf.onrender.com/investigate"
    payload = {
        "networks": [
            {
                "networkId": "prod-test-dict",
                "network": [
                    {"spy1": "Agent007", "spy2": "Agent008"},
                    {"spy1": "Agent008", "spy2": "Agent009"},
                    {"spy1": "Agent007", "spy2": "Agent009"}
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"Production Dict format - Status: {response.status_code}")
        print(f"Production Dict format - Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Production Dict format - Error: {e}")
        return False

def test_production_list_format():
    """Test production endpoint with list format"""
    url = "https://ubs-coding-challenge-5frf.onrender.com/investigate"
    payload = [
        {
            "networkId": "prod-test-list",
            "network": [
                {"spy1": "Agent007", "spy2": "Agent008"},
                {"spy1": "Agent008", "spy2": "Agent009"},
                {"spy1": "Agent007", "spy2": "Agent009"}
            ]
        }
    ]
    
    try:
        response = requests.post(url, json=payload)
        print(f"Production List format - Status: {response.status_code}")
        print(f"Production List format - Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Production List format - Error: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Investigate Endpoint Formats ===\n")
    
    print("Testing Local Endpoint:")
    print("-" * 40)
    dict_ok = test_dict_format()
    print()
    list_ok = test_list_format()
    print()
    
    print("Testing Production Endpoint:")
    print("-" * 40)
    prod_dict_ok = test_production_dict_format()
    print()
    prod_list_ok = test_production_list_format()
    print()
    
    print("Summary:")
    print("-" * 40)
    print(f"Local Dict Format: {'✅ PASS' if dict_ok else '❌ FAIL'}")
    print(f"Local List Format: {'✅ PASS' if list_ok else '❌ FAIL'}")
    print(f"Production Dict Format: {'✅ PASS' if prod_dict_ok else '❌ FAIL'}")
    print(f"Production List Format: {'✅ PASS' if prod_list_ok else '❌ FAIL'}")

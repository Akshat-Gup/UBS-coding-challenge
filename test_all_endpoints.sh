#!/bin/bash

# UBS Challenge API - Complete Endpoint Testing
# Use this script to test all endpoints locally or on the deployed server

BASE_URL="https://ubs-coding-challenge-5frf.onrender.com"
# For local testing, uncomment the line below:
# BASE_URL="http://localhost:8000"

echo "🚀 Testing UBS Challenge API at $BASE_URL"
echo "=================================================="

# Test 1: Health Check (GET)
echo "1️⃣ Testing Health Check (GET /)..."
curl -X GET "$BASE_URL/" -s | jq . || echo "Failed"
echo ""

# Test 2: Trivia (GET)
echo "2️⃣ Testing Trivia (GET /trivia)..."
curl -X GET "$BASE_URL/trivia" -s | jq . || echo "Failed"
echo ""

# Test 3: Ticketing Agent (POST)
echo "3️⃣ Testing Ticketing Agent (POST /ticketing-agent)..."
curl -X POST "$BASE_URL/ticketing-agent" \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "name": "CUSTOMER_A",
        "vip_status": false,
        "location": [1, 1],
        "credit_card": "CREDIT_CARD_1"
      }
    ],
    "concerts": [
      {
        "name": "CONCERT_1",
        "booking_center_location": [1, 5]
      }
    ],
    "priority": {
      "CREDIT_CARD_1": "CONCERT_1"
    }
  }' -s | jq . || echo "Failed"
echo ""

# Test 4: Blankety (POST)
echo "4️⃣ Testing Blankety (POST /blankety)..."
curl -X POST "$BASE_URL/blankety" \
  -H "Content-Type: application/json" \
  -d '{
    "series": [
      [0.1, null, 0.3, null, 0.5],
      [1.0, null, 0.9, 0.8, null]
    ]
  }' -s | jq '.answer[0][:5]' || echo "Failed"
echo ""

# Test 5: Trading Formula (POST) 
echo "5️⃣ Testing Trading Formula (POST /trading-formula)..."
curl -X POST "$BASE_URL/trading-formula" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "formula": "close > open",
      "data": [
        {"open": 100, "close": 105, "high": 110, "low": 95},
        {"open": 105, "close": 102, "high": 108, "low": 100}
      ]
    }
  ]' -s | jq . || echo "Failed"
echo ""

# Test 6: Investigate (POST) - Using dict format with "networks" key
echo "6️⃣ Testing Investigate (POST /investigate)..."
curl -X POST "$BASE_URL/investigate" \
  -H "Content-Type: application/json" \
  -d '{
    "networks": [
      {
        "networkId": "test-network",
        "network": [
          {"spy1": "Bruce Wayne", "spy2": "Diana Prince"},
          {"spy1": "Clark Kent", "spy2": "Diana Prince"},
          {"spy1": "Bruce Wayne", "spy2": "Clark Kent"}
        ]
      }
    ]
  }' -s | jq . || echo "Failed"
echo ""

echo "✅ All endpoint tests completed!"
echo "Check the responses above for any failures."

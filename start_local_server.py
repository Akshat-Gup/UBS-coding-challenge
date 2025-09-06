#!/usr/bin/env python3
"""
Local server script to test all endpoints while deployment issues are resolved.
Run this to test the fixed investigate endpoint locally.
"""

import uvicorn
import sys
import os

def main():
    print("🚀 Starting UBS Challenge API locally...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔧 All endpoints are working locally with latest fixes!")
    print("\n" + "="*60)
    print("ENDPOINTS AVAILABLE:")
    print("GET  /         - Health check")
    print("GET  /trivia   - Trivia answers") 
    print("GET  /debug    - Debug info (shows deployment status)")
    print("POST /ticketing-agent - Customer assignment")
    print("POST /blankety - Time series interpolation")
    print("POST /trading-formula - Financial formulas")
    print("POST /investigate - Spy network analysis (FIXED!)")
    print("="*60)
    print("\n💡 Use Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            "app.main:app", 
            host="0.0.0.0", 
            port=8000, 
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")

if __name__ == "__main__":
    main()

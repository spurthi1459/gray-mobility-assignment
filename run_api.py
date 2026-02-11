"""
Run FastAPI from project root
Usage: python run_api_simple.py
"""

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("STARTING GRAY MOBILITY SMART AMBULANCE API")
    print("=" * 60)
    print("\n✓ API will be available at: http://localhost:8000")
    print("✓ Interactive docs at: http://localhost:8000/docs")
    print("✓ Press Ctrl+C to stop\n")
    
    uvicorn.run(
        "src.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )
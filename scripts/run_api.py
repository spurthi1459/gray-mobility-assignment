"""
Run the FastAPI server

"""

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("STARTING GRAY MOBILITY SMART AMBULANCE API")
    print("=" * 60)
    print("\n API will be available at: http://localhost:8000")
    print("Interactive docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
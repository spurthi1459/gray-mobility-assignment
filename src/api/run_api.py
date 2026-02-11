"""
Run the FastAPI server

"""

import sys
import os

# Add project root to path BEFORE importing uvicorn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("STARTING GRAY MOBILITY SMART AMBULANCE API")
    print("=" * 60)
    print("\n API will be available at: http://localhost:8000")
    print(" Interactive docs at: http://localhost:8000/docs")
    
    
    # Change reload=False to avoid multiprocessing issues on Windows
    uvicorn.run(
        "src.api.app:app",
        host="127.0.0.1",  # Changed from 0.0.0.0
        port=8000,
        reload=False  # Changed from True (fixes Windows import issues)
    )
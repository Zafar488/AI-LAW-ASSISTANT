import os
import sys
from pathlib import Path

# Set the project root directory (this file is in lawyer_chatbot/)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

# Run the server
if __name__ == "__main__":
    try:
        import uvicorn
        from api.server import app
        print("Starting server on http://localhost:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
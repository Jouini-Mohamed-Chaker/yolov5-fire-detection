# app.py
import uvicorn
from fastapi import FastAPI
from .endpoints import router as api_router

app = FastAPI(
    title="Fire Detection System",
    description="Real-time fire detection system using YOLOv5",
    version="1.0.0"
)

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins – change if needed.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8026, reload=True)

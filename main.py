import os
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env only in development
# On Render, we use the dashboard environment variables
if os.getenv("RENDER") is None:
    load_dotenv()

app = FastAPI(
    title="STW AI API",
    description="Multi-model AI chat API using OpenRouter",
    version="1.0.0"
)

# 🔐 IMPORTANT: Configure CORS for your GitHub Pages domain
# Replace with your actual GitHub Pages URL
GITHUB_PAGES_URL = "https://stwanubhav.github.io/stwai"  # CHANGE THIS
YOUR_BACKEND_URL = "https://stw-ai-backend.onrender.com"  # Will be your Render URL

ALLOWED_ORIGINS = [
    GITHUB_PAGES_URL,
    f"https://*.stwanubhav.github.io",  # Wildcard for subdomains
    "http://localhost:3000",  # Local development
    "http://127.0.0.1:5500",  # VS Code Live Server
    "http://localhost:8000",  # Local FastAPI
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SITE_URL = os.getenv("SITE_URL", YOUR_BACKEND_URL)
SITE_NAME = os.getenv("SITE_NAME", "STW AI")

class ChatRequest(BaseModel):
    prompt: str
    models: List[str]

class ChatResponse(BaseModel):
    responses: Dict[str, str]
    status: str = "success"

class HealthCheck(BaseModel):
    status: str
    api_configured: bool
    environment: str

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "STW AI API is running",
        "docs": "/docs",
        "health": "/health",
        "environment": "production" if os.getenv("RENDER") else "development"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint - Render uses this to verify your app is running"""
    return HealthCheck(
        status="healthy",
        api_configured=bool(OPENROUTER_API_KEY),
        environment="production" if os.getenv("RENDER") else "development"
    )

@app.get("/api/models")
async def get_available_models():
    """Get list of recommended models"""
    return {
        "models": [
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-2",
            "google/palm-2",
            "meta-llama/llama-2-70b",
            "mistralai/mistral-7b"
        ]
    }

async def fetch_ai_response(client: httpx.AsyncClient, model: str, prompt: str) -> tuple:
    """Fetch response from a single model"""
    try:
        response = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30.0
        )
        
        response.raise_for_status()
        data = response.json()
        
        if 'choices' in data and len(data['choices']) > 0:
            content = data['choices'][0]['message']['content']
            return model, content
        else:
            return model, f"Error: Unexpected response format from {model}"
            
    except httpx.TimeoutException:
        return model, f"Error: Request timeout for model {model}"
    except httpx.HTTPStatusError as e:
        return model, f"Error: HTTP {e.response.status_code}"
    except Exception as e:
        return model, f"Error: {str(e)}"

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a prompt to multiple AI models and get responses
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="OpenRouter API key not configured"
        )
    
    if not request.models:
        raise HTTPException(
            status_code=400,
            detail="At least one model must be specified"
        )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                fetch_ai_response(client, model, request.prompt) 
                for model in request.models
            ]
            results = await asyncio.gather(*tasks)

        response_data = {model: text for model, text in results}
        return ChatResponse(responses=response_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

# STW AI Backend API

FastAPI backend for STW AI application, deployed on Render.

## 🚀 Live API

Base URL: `https://stw-ai-backend.onrender.com`

## 📚 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/api/models` | GET | List available models |
| `/api/chat` | POST | Send prompt to multiple models |

## 🔧 Local Development

1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

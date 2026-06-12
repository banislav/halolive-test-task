# Halolive Deep Agents

LangChain-based deep agent architecture prototype with a FastAPI runtime bridge and a
TanStack/Vite frontend.

## Docker Compose Development

Create a root `.env` with one of the OpenRouter keys:

```text
OPENROUTER_API_KEY=...
```

or:

```text
DEEP_AGENTS_OPENROUTER_API_KEY=...
```

Start the real backend and frontend together:

```bash
docker compose up --build
```

Open the app:

```text
http://localhost:5173
```

Backend API docs:

```text
http://localhost:8000/docs
```

The Compose setup defaults the frontend to real backend mode:

```text
VITE_API_BASE=http://localhost:8000
VITE_MOCK_MODE=false
```

Runtime sessions are in-memory in this development setup.

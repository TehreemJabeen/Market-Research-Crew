# Stage 1: build the React app
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

# Stage 2: API + built frontend
FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Optional local sentiment model (large: torch). Build with: docker build --build-arg WITH_ML=1 ...
# Without it the app uses the LLM scorer automatically.
ARG WITH_ML=0
RUN if [ "$WITH_ML" = "1" ]; then pip install --no-cache-dir -r requirements-ml.txt; fi

COPY app ./app
COPY api ./api
COPY ui ./ui
COPY evals ./evals
COPY --from=web /web/dist ./frontend/dist

ENV RUNS_DB=/data/runs.db
VOLUME /data
EXPOSE 8000
# docker run --env-file .env -p 8000:8000 -v crew-data:/data market-research-crew
# Demo without keys: add -e DEMO_MODE=1. Streamlit still works: override CMD with
#   streamlit run ui/streamlit_app.py --server.address=0.0.0.0
CMD ["uvicorn", "api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

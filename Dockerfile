FROM python:3.12-slim

WORKDIR /srv
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Optional local sentiment model (large: torch). Build with: docker build --build-arg WITH_ML=1 ...
# Without it the app uses the LLM scorer automatically.
ARG WITH_ML=0
RUN if [ "$WITH_ML" = "1" ]; then pip install --no-cache-dir -r requirements-ml.txt; fi

COPY app ./app
COPY ui ./ui
COPY evals ./evals

EXPOSE 8501
# Pass keys at runtime: docker run --env-file .env -p 8501:8501 market-research-crew
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

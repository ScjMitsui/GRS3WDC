FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir .

COPY experiments ./experiments
COPY configs ./configs
COPY data ./data

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "experiments.run_all"]

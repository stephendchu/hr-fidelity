FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY pyproject.toml .
RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30"

# Copy source and required data
COPY src/ src/
COPY data/reqs/ data/reqs/
COPY data/names/ data/names/

# Install the package itself
RUN pip install --no-cache-dir -e .

# Design F is the production skin
ENV HRFIDELITY_INDEX=designs/design-f-landing.html

EXPOSE 8000

CMD python -m hrfidelity serve --host 0.0.0.0 --port ${PORT:-8000}

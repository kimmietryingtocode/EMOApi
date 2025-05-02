# ------------ Stage 1: Builder ------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies to /install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt

# ------------ Stage 2: Final image ------------
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages and source code
COPY --from=builder /install /usr/local
COPY ./app ./app
COPY ./Makefile ./Makefile

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

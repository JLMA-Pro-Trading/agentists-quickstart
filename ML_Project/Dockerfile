# Enhanced Universal Trading Syndicate - Production Docker Image
# Educational Trading System with Multi-Stage Build

FROM python:3.11-slim-bullseye as builder

# Set build arguments
ARG BUILDPLATFORM
ARG TARGETPLATFORM

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    wget \
    curl \
    git \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib (Technical Analysis Library)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRADING_MODE=paper \
    SECURITY_LEVEL=educational \
    TZ=UTC

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy TA-Lib from builder
COPY --from=builder /usr/lib/libta_lib.* /usr/lib/
COPY --from=builder /usr/include/ta-lib/ /usr/include/ta-lib/

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create application user
RUN groupadd -r trading && useradd -r -g trading trading

# Create application directories
RUN mkdir -p /app/logs /app/data /app/vault && \
    chown -R trading:trading /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=trading:trading . .

# Create educational trading constraints file
RUN echo "# EDUCATIONAL TRADING CONSTRAINTS" > /app/.educational_mode && \
    echo "LIVE_TRADING_DISABLED=true" >> /app/.educational_mode && \
    echo "PAPER_TRADING_ONLY=true" >> /app/.educational_mode && \
    echo "BINANCE_TESTNET_ONLY=true" >> /app/.educational_mode && \
    chown trading:trading /app/.educational_mode && \
    chmod 444 /app/.educational_mode

# Switch to non-root user
USER trading

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=10)"

# Expose ports
EXPOSE 8000 8001 8080

# Default command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
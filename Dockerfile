FROM python:3.12-slim

RUN useradd -m -u 1000 user

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ca-certificates \
    dnsutils \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

USER user

EXPOSE 7860

CMD ["streamlit", "run", "streamlit/app.py", "--server.address=0.0.0.0", "--server.port=7860"]
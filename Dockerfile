FROM python:3.12

RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

USER user

EXPOSE 7860

CMD ["streamlit", "run", "streamlit/app.py", "--server.address=0.0.0.0", "--server.port=7860"]
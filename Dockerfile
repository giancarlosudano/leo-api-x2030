# Usa l'immagine ufficiale di Python come base
FROM python:3.9-slim

# Imposta la directory di lavoro
WORKDIR /app

# Copia il file dei requisiti e installa le dipendenze
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia il codice dell'applicazione
COPY app.py app.py

# Espone la porta su cui Flask sta in ascolto
EXPOSE 5000

# Comando per eseguire l'applicazione
CMD ["python", "app.py"]
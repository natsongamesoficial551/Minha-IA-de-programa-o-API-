# Usando Python 3.11 estável
FROM python:3.11-slim

# Diretório de trabalho dentro do container
WORKDIR /app

# Copia requirements.txt
COPY requirements.txt .

# Instala dependências do sistema que podem ser necessárias
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências do Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copia todo o código do projeto
COPY . .

# Expõe a porta para o FastAPI/Uvicorn
EXPOSE 8000

# Comando para rodar a API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

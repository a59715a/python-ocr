# 使用Python 3.11作為基礎映像
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 複製需要的檔案
COPY main.py .
COPY requirements.txt .

# 安裝依賴
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 8000

# 啟動應用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 
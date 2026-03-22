# Sử dụng Python 3.11 slim để tương thích tốt nhất với các lib AI (faiss, sentence-transformers)
FROM python:3.11-slim

# Thiết lập môi trường không buffer để log in thẳng ra terminal
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Cài đặt các dependency hệ thống cần cho quá trình build C++ library (nếu có)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements vào trước để tận dụng Docker Cache
COPY requirement.txt .

# Nâng cấp pip và cài đặt toàn bộ libs
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirement.txt

# Khởi tạo một file version rỗng mặc định nếu chưa tồn tại
RUN echo "1.0.0" > keyword_list_version.txt

# Copy toàn bộ mã nguồn và Database CSV
COPY test.py .
COPY data.csv .

# Chạy trực tiếp qua Python CLI
CMD ["python", "test.py"]

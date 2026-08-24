FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY data ./data
COPY tests ./tests
RUN pip install --no-cache-dir -e '.[dev]'
CMD ["pytest", "-q"]

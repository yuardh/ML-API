# Stage 1: Build app
FROM python:3.10-slim AS build
ENV PYTHONBUFFERED True

COPY requirements.txt /
RUN pip install --no-cache-dir -r requirements.txt

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Stage 2: Create final image
FROM python:3.10-slim
ENV PYTHONBUFFERED True

COPY --from=build /app /app
WORKDIR /app

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app

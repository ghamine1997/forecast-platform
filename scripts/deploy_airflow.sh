#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

TAG=$(date +%Y%m%d-%H%M%S)
echo "Building forecast-airflow:$TAG"
docker build -f docker/airflow.Dockerfile -t forecast-airflow:$TAG .

echo "Loading image into kind"
kind load docker-image forecast-airflow:$TAG --name forecast

echo "Deploying with Helm"
helm upgrade --install airflow apache-airflow/airflow \
  -n forecast -f helm/airflow-values.yaml \
  --set images.airflow.tag=$TAG \
  --timeout 15m
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

TAG=$(date +%Y%m%d-%H%M%S)
echo "Building forecast-dashboard:$TAG"
docker build -f docker/dashboard.Dockerfile -t forecast-dashboard:$TAG .

echo "Loading image into kind"
kind load docker-image forecast-dashboard:$TAG --name forecast

echo "Deploying"
sed "s|forecast-dashboard:TAG|forecast-dashboard:$TAG|" k8s/streamlit.yaml | kubectl apply -f -
kubectl rollout status deployment/dashboard -n forecast --timeout=180s

echo "Dashboard ready on port 30501"
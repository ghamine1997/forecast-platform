# Runbook: Forecasting Platform

Day-to-day operations and troubleshooting. All commands run from the project root (`~/forecast-platform`) in the VS Code terminal connected to the VM.

## Stop the platform

```bash
git status                              # commit and push anything pending
docker stop forecast-control-plane      # stops the whole cluster, data is kept
sudo shutdown now
```

## Start the platform

1. Start the VM in VMware and wait for the login screen.
2. Connect VS Code: `Ctrl+Shift+P` → **Remote-SSH: Connect to Host** → `amine@<VM_IP>`.
   If the connection fails, log in on the VM window and run `hostname -I` to get the new IP.
3. Start the cluster and check it:

```bash
docker start forecast-control-plane
sleep 120
kubectl get nodes
kubectl get pods -n forecast
```

Airflow pods may restart a few times in the first minutes while PostgreSQL starts. If a pod is still failing after 5 minutes: `kubectl delete pod -n forecast <pod-name>`.

## Service URLs

| Service | URL |
|---|---|
| Airflow | `http://<VM_IP>:30080` (admin / admin) |
| Dashboard | `http://<VM_IP>:30501` |
| MinIO console | `http://<VM_IP>:30901` (minioadmin / minioadmin123) |

## Deploy a code change

| Changed | Command |
|---|---|
| `dags/` or `src/` (pipeline) | `./scripts/deploy_airflow.sh` |
| `dashboard/` (or `src/config.py`, `src/metrics.py`) | `./scripts/deploy_dashboard.sh` |
| `k8s/postgres.yaml` or `k8s/minio.yaml` | `kubectl apply -f k8s/<file>.yaml` |
| `helm/airflow-values.yaml` | `./scripts/deploy_airflow.sh` |

Before deploying Airflow, pause `forecasting_pipeline` in the UI (toggle next to its name) so no scheduled run starts while pods are being replaced. Unpause once all pods are `Running`.

## Run the pipeline

- Automatic: every Monday at 02:00 (only if the VM and cluster are running).
- Manual: Airflow UI → `forecasting_pipeline` → **Trigger**. A full run takes about 10–12 minutes.
- Run one stage locally for debugging:

```bash
source .venv/bin/activate
python -m src.ingest        # or clean, features, tune 3, train, cross_validate, evaluate, register, forecast, publish
```

## Health checks

```bash
kubectl get pods -n forecast                       # all Running or Completed
free -h                                            # memory and swap
df -h /                                            # disk space
kubectl exec -it -n forecast deploy/postgres -- psql -U forecast -d forecast -P pager=off -c "SELECT version, model_name, stage, test_smape FROM model_registry ORDER BY version DESC LIMIT 3;"
kubectl exec -it -n forecast deploy/postgres -- psql -U forecast -d forecast -P pager=off -c "SELECT * FROM data_quality_log ORDER BY run_at DESC LIMIT 5;"
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| VS Code cannot connect | VM IP changed | Run `hostname -I` on the VM, reconnect with the new IP |
| Browser page does not load | Cluster stopped, or pod not ready | `docker start forecast-control-plane`, then `kubectl get pods -n forecast` |
| Pod in `ImagePullBackOff` | Image name not found in the Kind node | `kubectl describe pod -n forecast <pod>`; re-run the deploy script |
| Pod in `CrashLoopBackOff` | App error at startup | `kubectl logs -n forecast <pod>` (add `-c <container>` if asked) |
| Airflow task `Up For Retry` | Temporary error, often pods replaced during a deploy | Wait for the retry (5 min); if it fails again, read the task log |
| Airflow task failed in `register_model` | Quality gate: test SMAPE above 10% | Check `evaluation_results`; investigate data or features before re-running |
| `No production model yet` on the dashboard | Pipeline never completed | Trigger `forecasting_pipeline` and wait for all tasks to finish |
| Dashboard shows old numbers | 10-minute cache | Click **Refresh data** in the sidebar |
| `ModuleNotFoundError` when running scripts locally | Virtual environment not active | `source .venv/bin/activate` |
| Disk almost full | Old images from deployments | `docker image prune -a` and `docker exec forecast-control-plane crictl rmi --prune` |
| Terminal stuck showing `(END)` | psql pager | Press `q`, or add `-P pager=off` to the command |

A NodePort service is reachable only if three numbers agree: the app's listening port equals the Service `targetPort`, the Service `nodePort` equals the `containerPort` in `k8s/kind-config.yaml`, and the browser uses that mapping's `hostPort`.

## Useful commands

```bash
kubectl get svc -n forecast                 # services and ports
kubectl get pvc -n forecast                 # persistent disks
helm list -n forecast                       # Helm releases
kubectl logs -n forecast deploy/dashboard   # dashboard logs
docker exec forecast-control-plane crictl images | grep forecast   # images inside the node
```

## Full reset (destroys all data)

```bash
kind delete cluster --name forecast
```

Then follow **Getting started** in the README to rebuild from scratch.

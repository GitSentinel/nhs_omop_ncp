import mlflow

mlflow.set_tracking_uri('sqlite:///mlflow_runs/mlflow.db')
client = mlflow.MlflowClient()

exp = client.get_experiment_by_name('nhs_omop_agent')
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string='',
)
print(f'Total runs in DB: {len(runs)}')
for r in runs:
    q = r.data.params.get('query', 'N/A')[:50]
    print(f'  {r.info.run_name:<20} status={r.info.status:<10} query={q}')
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://0.0.0.0:5001")
mlflow_client = MlflowClient()

# Get the experiment by name
experiment = mlflow_client.get_experiment_by_name("wine_dataset")

if experiment is not None:
    experiment_id = experiment.experiment_id
    if runs := mlflow_client.search_runs(experiment_ids=[experiment_id]):
        # # Sorting the runs by their metrics.rmse in ascending order
        # sorted_runs = sorted(
        #     runs, key=lambda run: run.data.metrics["validation-mlogloss"]
        # )
        
        # Sorting the runs by their metrics.rmse in ascending order
        sorted_runs = sorted(
            runs, key=lambda run: run.info.start_time, reverse=True
        )

        # Get the best run (the one with the lowest rmse)
        best_run = sorted_runs[0]

        model_name = "BestWineDatasetModel"
        # Check if the model is already registered. If not, register.
        if model_name not in [rm.name for rm in mlflow.search_registered_models()]:
            mlflow_client.create_registered_model(name=model_name)

        # Register the model from the best run
        result = mlflow_client.create_model_version(
            name=model_name,
            source=f"{best_run.info.artifact_uri}/model",
            run_id=best_run.info.run_id,
        )
        print(f"Registered model version: {result.version}")

        version = result.version  # version from the previous step
        stage = "Production"  # or "Staging" depending on your use case
        mlflow_client.transition_model_version_stage(
            name=model_name, version=version, stage=stage
        )
        print(f"Transitioned model version: {version} to '{stage}' stage.")

    else:
        print(f"No runs found for experiment_id {experiment_id}")

else:
    print("No experiment with the name 'wine_dataset' found")

import os
from datetime import datetime
from sklearn.model_selection import train_test_split
import optuna

n_split = 5
output_path = "Output"
output_mark = f"Optuna_Cu_without_metal_features"

X_train, X_val, y_train, y_val = train_test_split(X, y)


def objective(trial: optuna.Trial):
    conv_features = trial.suggest_categorical("conv_features", [75])
    model = model(model_parameters)
    model.fit(X_train, y_train)
    return model.score(X_val, y_val)


def callback(study: optuna.Study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="metrics", value=trial.user_attrs["metrics"])


if __name__ == "__main__":
    start = datetime.now()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1, timeout=None, callbacks=[callback])
    end = datetime.now()

    best_params = {}
    result = {
        "trials": len(study.trials),
        "started": str(start).split(".")[0],
        "finished": str(end).split(".")[0],
        "duration": str(end - start).split(".")[0],
        "train_r2_score": study.user_attrs["metrics"]["train_r2_score"],
        "valid_r2_score": study.user_attrs["metrics"]["valid_r2_score"],
        "test_r2_score": study.user_attrs["metrics"]["test_r2_score"],
        "train_mean_squared_error": study.user_attrs["metrics"]["train_mean_squared_error"],
        "valid_mean_squared_error": study.user_attrs["metrics"]["valid_mean_squared_error"],
        "test_mean_squared_error": study.user_attrs["metrics"]["test_mean_squared_error"],
        "train_mean_absolute_error": study.user_attrs["metrics"]["train_mean_absolute_error"],
        "valid_mean_absolute_error": study.user_attrs["metrics"]["valid_mean_absolute_error"],
        "test_mean_absolute_error": study.user_attrs["metrics"]["test_mean_absolute_error"],
        "model_parameters": best_params,
    }

    time_mark = str(start).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

    study.trials_dataframe().to_csv(path_or_buf=os.path.join(output_path, f"{output_mark}_{time_mark}.csv"))
    with open(os.path.join(output_path, f"{output_mark}_{time_mark}.json"), "w") as jf:
        json.dump(result, jf)

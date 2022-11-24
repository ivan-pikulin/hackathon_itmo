import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
import optuna
from torch import nn
from torch.optim import Adam

from Source.models import FCNN
from Source.trainer import ModelTrainer

n_split = 5
output_path = "Output"
output_mark = f"Optuna"

folds, test_dl = train_test_valid_split(dataset, n_splits=5, test_ratio=0.2, batch_size=64)


def objective(trial: optuna.Trial):
    n_layers = trial.suggest_int("n_layers", 1, 7)
    hidden_dims = [trial.suggest_int(f"hidden_{i}", 64, 256, 64) for i in range(n_layers - 1)]
    dropout = trial.suggest_float("dropout", 0, 0.5)
    bn = trial.suggest_categorical("bn", [True, False])

    actf_variants = {
        "nn.ELU()": nn.ELU(),
        "nn.LeakyReLU()": nn.LeakyReLU(),
        "nn.LogSigmoid()": nn.LogSigmoid(),
        "nn.PReLU()": nn.PReLU(),
        "nn.ReLU()": nn.ReLU(),
        "nn.ReLU6()": nn.ReLU6(),
        "nn.RReLU()": nn.RReLU(),
        "nn.SELU()": nn.SELU(),
        "nn.CELU()": nn.CELU(),
        "nn.GELU()": nn.GELU(),
        "nn.SiLU()": nn.SiLU(),
        "nn.Mish()": nn.Mish(),
        "nn.Softplus()": nn.Softplus(),
        "nn.Softshrink()": nn.Softshrink(),
        "nn.Softsign()": nn.Softsign(),
        "nn.Tanh()": nn.Tanh(),
        "nn.Tanhshrink()": nn.Tanhshrink(),
    }
    actf_name = trial.suggest_categorical("actf", list(actf_variants.values()))
    actf = actf_variants[actf_name]

    optimizer_parameters = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2),
        "betas": (
            trial.suggest_float("betas_0", 0, 1),
            trial.suggest_float("betas_1", 0, 1),
        )
    }

    model_parameters = {
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "bn": bn,
        "actf": actf,
        "optimizer": Adam,
        "optimizer_parameters": optimizer_parameters,
    }

    model = FCNN(**model_parameters)
    trainer = ModelTrainer(
        model=model,
        train_valid_data=folds,
        test_data=test_dl,
        output_folder=None,
        out_folder_mark=None,
        es_patience=100,
        epochs=1000,
        verbose=True
    )

    metrics = trainer.train_cv_models()
    trial.set_user_attr(key="metrics", value=metrics)

    return metrics["valid_mean_squared_error"]


def callback(study: optuna.Study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="metrics", value=trial.user_attrs["metrics"])

        n_layers = int(trial.params["n_layers"])
        best_params = {
            "n_layers": n_layers,
            "hidden_dims": [75] + [trial.params[f"hidden_{i}"] for i in range(n_layers - 1)],
            "dropout": trial.params["dropout"],
            "bn": trial.params["bn"],
            "actf": trial.params["actf"],
            "optimizer_parameters": {
                "lr": trial.params["lr"],
                "betas": (
                    trial.params["betas_0"],
                    trial.params["betas_1"],
                ),
            }

        }
        study.set_user_attr(key="best_params", value=best_params)


if __name__ == "__main__":
    start = datetime.now()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1, timeout=None, callbacks=[callback])
    end = datetime.now()

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
        "model_parameters": study.user_attrs["best_params"],
    }

    time_mark = str(start).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

    study.trials_dataframe().to_csv(path_or_buf=os.path.join(output_path, f"{output_mark}_{time_mark}.csv"))
    with open(os.path.join(output_path, f"{output_mark}_{time_mark}.json"), "w") as jf:
        json.dump(result, jf)

import copy
import json
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, confusion_matrix, precision_score, \
    matthews_corrcoef, f1_score, recall_score, accuracy_score, roc_auc_score
from itertools import chain
import os
from datetime import datetime
from pytorch_lightning import loggers as pl_loggers

import numpy as np
import sys

sys.path.append("./Source")

CLASS_METRICS = {"confusion_matrix": confusion_matrix, "f1_score": f1_score, "roc_auc_score": roc_auc_score,
                 "precision_score": precision_score, "recall_score": recall_score,
                 "matthews_corrcoef": matthews_corrcoef, "accuracy_score": accuracy_score}
REG_METRICS = {"r2_score": r2_score, "mean_squared_error": mean_squared_error,
               "mean_absolute_error": mean_absolute_error}


def get_best_fold(pretrained_folder):
    """
    Gets index of best fold according to validation loss

    Parameters
    ----------
    pretrained_folder : str
        Path to folder with pretrained model

    Returns
    -------
    best_fold : int
        Index of fold with the lowest validation loss
    """
    fold_dirs = [os.path.join(pretrained_folder, i) for i in os.listdir(pretrained_folder) if i.startswith("fold")]
    best_loss_dict = {}
    for i, fold in enumerate(fold_dirs):
        with open(os.path.join(pretrained_folder, fold, "losses.json")) as jf:
            jd = json.load(jf)
        best_loss_dict[min(jd["valid_loss"])] = i

    return best_loss_dict[min(list(best_loss_dict.keys()))]


def calculate_metrics(train_true, train_pred, valid_true, valid_pred, test_true=None, test_pred=None,
                      mode="regression"):
    results = {}

    if mode == "classification":
        for metric in CLASS_METRICS:
            results["{}_{}".format("train", metric)] = float(CLASS_METRICS[metric](train_true,
                                                                                   np.argmax(train_pred, axis=-1)))
            results["{}_{}".format("valid", metric)] = float(CLASS_METRICS[metric](valid_true, np.argmax(valid_pred,
                                                                                                         axis=-1)))
            if test_true is not None:
                results["{}_{}".format("test", metric)] = float(CLASS_METRICS[metric](test_true,
                                                                                      np.argmax(test_pred,
                                                                                                axis=-1)))
    elif mode == "regression":
        for metric in REG_METRICS:
            results["{}_{}".format("train", metric)] = float(REG_METRICS[metric](train_true, train_pred))
            results["{}_{}".format("valid", metric)] = float(REG_METRICS[metric](valid_true, valid_pred))
            if test_true is not None:
                results["{}_{}".format("test", metric)] = float(REG_METRICS[metric](test_true, test_pred))

    for key in results.keys():
        if "confusion" in key:
            results[key] = results[key].tolist()

    return results


class ModelTrainer:
    """
    Class for cross-validation training of models

    Attributes
    ----------
    model : <class 'torch.nn.Module'>
        Class of models to be trained on different folds
    models : list
        List of n_split models trained on different folds
    mode : str
        Type of problem. May be "regression" or "classification".
    n_split : int
        Number of folds in KFold cross-validation
    train_valid_data : list
        list of n_split tuples of dataloaders like: [(fold_1_train_dl, fold_1_val_dl), (fold_2_train_dl, fold_2_val_dl), ...]
    test_data : Dataset
    output_folder : str
        Path to folder where all output data about training process should be located
    out_folder_mark : str
        Mark which will be included to name of output folder for current training process
    main_folder : str
        Path to folder where output of current training process will be located
    es_patience : int
        Patience parameter for EarlyStopping
    epochs : int
        Number of learning epochs

    Methods
    ----------
    prepare_out_folder
        Create directory for current traing process and subdirectories for every fold
    write_model_structure
        Write model structure to file "model_structure.json" in self.main_folder
    get_full_train_loader
        Returns DataLoader with all data (from all folds)
    train_cv_models(pretrained_folder=False, layers_to_freeze=None)
        Create model, train it with KFold cross-validation and write down metrics
    train_model(model, train_fold, valid_fold, fold_num, epochs=1000)
        Train model on certain fold and write down metrics
    single_fold_predict(model, features)
        Make prediction on a single fold
    make_predictions(features: DataLoader)
        Make prediction by all trained models
    """

    def __init__(self, model, train_valid_data, test_data=None, output_folder=None, out_folder_mark=None,
                 es_patience=100, epochs=1000, verbose=True):
        self.initial_model = model
        self.models = []
        self.train_valid_data = train_valid_data
        self.test_data = test_data
        self.output_folder = output_folder
        self.out_folder_mark = out_folder_mark
        self.es_patience = es_patience
        self.n_split = len(self.train_valid_data)
        self.epochs = epochs
        self.verbose = verbose

        time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]
        self.main_folder = os.path.join(self.output_folder, f"{self.out_folder_mark}_{time_mark}")
        if self.verbose:
            self.prepare_out_folder()

    def prepare_out_folder(self):
        os.mkdir(self.main_folder)
        for fold in range(self.n_split):
            os.mkdir(os.path.join(self.main_folder, f"fold_{fold + 1}"))
        self.write_model_structure()

    def write_model_structure(self):
        with open(os.path.join(self.main_folder, "model_structure.json"), "w") as jf:
            structure = self.initial_model.get_model_structure()
            json.dump(structure, jf)

    def get_full_train_loader(self):
        train_loader, valid_loader = self.train_valid_data[0]
        full_train_list = train_loader.dataset + valid_loader.dataset

        return DataLoader(full_train_list)

    def train_cv_models(self, pretrained_folder=False, layers_to_freeze=None):
        """
        Create model, train it with KFold cross-validation and write down metrics

        Parameters
        ----------
        pretrained_folder : str, optional
            Path to pretrained model to be loaded and frozen according to layers_to_freeze param
        layers_to_freeze : list, optional
            List of layers to be frozen
        """
        for i, (train_fold, valid_fold) in enumerate(self.train_valid_data):
            model = copy.deepcopy(self.initial_model)
            if pretrained_folder:
                best_fold = get_best_fold(pretrained_folder)
                model.load_state_dict(
                    torch.load(os.path.join(pretrained_folder, f"fold_{best_fold + 1}", "best_model"))
                )
                model.freeze_layers(layers_to_freeze)

            self.train_model(model, train_fold, valid_fold, i, self.epochs)

        mean_valid_pred = np.concatenate(np.array([self.single_fold_predict(m, self.train_valid_data[i][1])
                                                   for i, m in enumerate(self.models)], dtype=object), axis=None)
        valid_true = np.concatenate(np.array([y for fold in self.train_valid_data for (x, y) in fold[1]],
                                             dtype=object))
        full_train_loader = self.get_full_train_loader()
        mean_train_pred = np.mean(np.array([self.single_fold_predict(m, full_train_loader)
                                            for i, m in enumerate(self.models)]), axis=0)
        train_true = np.array([y for (x, y) in full_train_loader])

        if self.test_data:
            mean_test_pred = np.mean(np.array([self.single_fold_predict(m, self.test_data) for m in self.models]),
                                     axis=0)
            test_true = np.concatenate(np.array([y for (x, y) in self.test_data],
                                                dtype=object))
            results_dict = calculate_metrics(train_true, mean_train_pred, valid_true, mean_valid_pred, test_true,
                                             mean_test_pred)
        else:
            results_dict = calculate_metrics(train_true, mean_train_pred, valid_true, mean_valid_pred)

        if self.verbose:
            with open(os.path.join(self.main_folder, "metrics.json"), "w") as jf:
                json.dump(results_dict, jf)
        return results_dict

    def train_model(self, model, train_fold, valid_fold, fold_num, epochs=1000):
        """
        Train model on certain fold and write down metrics

        Parameters
        ----------
        model : model
            Model to be trained
        train_fold : Dataset
        valid_fold : Dataset
        fold_num : int
            Number of KFold cross-validation folds
        epochs : int, optional
            Number of training epochs. Default is 1000.
        """
        if self.verbose:
            tb_logger = pl_loggers.TensorBoardLogger(os.path.join(self.main_folder, f"fold_{fold_num + 1}", "logs"))
        else:
            tb_logger = False
        es_callback = EarlyStopping(patience=self.es_patience, monitor="val_loss")

        trainer = Trainer(callbacks=[es_callback], log_every_n_steps=20, max_epochs=epochs, logger=tb_logger,
                          accelerator="auto")
        trainer.fit(model, train_fold, valid_fold)
        self.models.append(model)

        current_folder = os.path.join(self.main_folder, f"fold_{fold_num + 1}")
        torch.save(model.state_dict(), os.path.join(current_folder, "best_model"))
        with open(os.path.join(current_folder, "losses.json"), "w") as jf:
            json.dump({"train_loss": model.train_losses,
                       "valid_loss": model.valid_losses}, jf)

        train_predicted, valid_predicted = self.single_fold_predict(model, train_fold), \
                                           self.single_fold_predict(model, valid_fold)

        train_true = np.array([y for (x, y) in train_fold])
        valid_true = np.array([y for (x, y) in valid_fold])

        if self.test_data:
            test_predicted = self.single_fold_predict(model, self.test_data)
            test_true = np.array([y for (x, y) in self.test_data])
            results_dict = calculate_metrics(train_true, train_predicted, valid_true, valid_predicted, test_true,
                                             test_predicted)
        else:
            results_dict = calculate_metrics(train_true, train_predicted, valid_true, valid_predicted)

        with open(os.path.join(current_folder, "metrics.json"), "w") as jf:
            json.dump(results_dict, jf)

    @staticmethod
    def single_fold_predict(model, dataloader: DataLoader):
        """
        Make prediction on a single fold

        Parameters
        ----------
        model : model
            Model used for prediction
        features : DataLoader
            Features used for prediction

        Returns
        ----------
        out : numpy.array
            A vector of model predictions
        """
        predictions = []
        for batch in dataloader:
            x, y = batch
            predictions += list(model.forward(x).detach().numpy().flatten())
        return predictions

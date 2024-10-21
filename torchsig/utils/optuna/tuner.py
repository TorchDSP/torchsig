"""Optuna Optimizer classes

Example
    >>> from torchsig.utils.optuna.tuner import OptunaOptimizer
    override the `objective()` function
    >>> opt = YourOptunaOptimizer(n_trials=5, epochs=10, name="DocString Test")
    >>> opt.run_optimization()
"""


import torch
from torch.utils.data import DataLoader
import os
import yaml
import pytorch_lightning as pl
import yaml
import optuna
import copy
from torchsig.utils.yolo_train import Yolo_Trainer
from typing import Tuple, Union


class OptunaOptimizer():
    """Optuna Optimizer abstract base class

    Runs Optuna optimization workflow. Requires subclasses to override the `objective` function.

    Args:
        n_trials (int, optional): Number of hyperparameter trials to run. Defaults to 5.
        epochs (int, optional): Number of epochs to run per trial. Defaults to 5.
        name (str, optional): Name of optuna optimization session. Defaults to "Test".
    """

    def __init__(self, n_trials:int=5, epochs:int=5, name:str="Test"):
        self.n_trials = n_trials
        self.epochs = epochs
        self.name = name
        self.best_params = None

    def run_optimization(self, ret_params: bool = True) -> Union[optuna.study.Study, Tuple[optuna.study.Study, dict]]:
        """Runs Optuna optimization flow. Includes:
        - creating pruners
        - creating optuna study
        - running the optimization
        - printing out best trial and params

        Args:
            ret_params (bool, optional): Whether to return best parameters found. Defaults to True.

        Returns:
            optuna.study.Study | Tuple[optuna.study.Study, dict]: Returns optuna Study and optionally a dictionary of best parameters.
        """
        pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=self.epochs)
        study = optuna.create_study(study_name=self.name, direction="minimize", pruner=pruner)
        study.optimize(self.objective, n_trials=self.n_trials)

        print("\n\nNumber of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        self.best_params = study.best_params
            
        if ret_params:
            return study, trial.params
        else:
            return study

    def objective(self, trial: optuna.trial.Trial) -> float:
        """Optuna Objective function to optimize. Runs once per trial.

        Args:
            trial (optuna.trial.Trial): Current trial.

        Raises:
            NotImplementedError: Subclasses must implement this method.

        Returns:
            float: Score to evaluate performance. Usually set to validation loss.
        """
        raise NotImplementedError("Define optuna objective function.")


class YoloOptunaOptimizer(OptunaOptimizer):
    """YOLO Optuna Optimizer

    Args:
            overrides (dict): YOLO overrides dictionary.
    """

    def __init__(self, overrides: dict, **kwargs):
        super().__init__(**kwargs)

        self.original_overrides = copy.deepcopy(overrides)
        self.overrides = overrides
        self.overrides['verbose'] = False
        self.overrides['epochs'] = self.epochs
        self.overrides['save'] = False
        #self.overrides['device'] = 0
        # self.overrides['workers'] = len(os.sched_getaffinity(0)) - 1
        self.overrides['workers'] = 1

    def objective(self, trial: optuna.trial.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-2)
        cos_lr = trial.suggest_categorical("cos_lr", [False, True])
        freeze = trial.suggest_int("freeze", 0, 5)
        imgsz_power2 = trial.suggest_int("imgsz_power2", 6, 9) # 2^6=64 to 2^9=512
        imgsz = 2 ** imgsz_power2
        optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])

        trial_overrides = copy.deepcopy(self.overrides)
        trial_overrides['lr0'] = lr
        trial_overrides['cos_lr'] = cos_lr
        trial_overrides['freeze'] = freeze
        trial_overrides['imgsz'] = imgsz
        trial_overrides['optimizer'] = optimizer
        

        trainer = Yolo_Trainer(overrides=trial_overrides)

        trainer.train()

        return trainer.fitness

    def get_optimized_overrides(self) -> dict:
        """Returns YOLO dictionary with best parameters set.

        Returns:
            dict: YOLO overrides dict with best params.
        """
        overrides_optimized = self.original_overrides
        for k in self.best_params.keys():
            if k =="imgsz_power2":
                overrides_optimized["imgsz"] = 2 ** self.best_params[k]
            elif k == "lr":
                overrides_optimized["lr0"] = self.best_params[k]
            else:
                overrides_optimized[k] = self.best_params[k]

        return overrides_optimized

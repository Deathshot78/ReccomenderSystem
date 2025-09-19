import optuna
import numpy as np
import math
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import implicit
from utils import prepare_ground_truth, calculate_metrics
from models import recommend_als_and_evaluate
from data_prepare import prepare_data

def objective_als(trial, train_df, val_df):
    """
    The objective function for Optuna to optimize.
    """
    # 1. Define the hyperparameter search space
    params = {
        'factors': trial.suggest_int('factors', 20, 200),
        'regularization': trial.suggest_float('regularization', 1e-3, 1e-1, log=True),
        'iterations': trial.suggest_int('iterations', 10, 50)
    }
    
    # 2. Run an evaluation with the suggested parameters
    metrics = recommend_als_and_evaluate(train_df, val_df, **params)
    
    # 3. Return the metric we want to maximize (precision)
    return metrics['mean_precision@k']

def tune_als_hyperparameters(train_df, val_df, n_trials=25):
    """
    Orchestrates the Optuna study to find the best hyperparameters for ALS.
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_als(trial, train_df, val_df), n_trials=n_trials)
    
    print("\n--- Optuna Study Complete ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Precision@10): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    return trial.params

if __name__ == "__main__":
    # 1. Prepare all data
    train_set, validation_set, test_set = prepare_data()

    # --- Hyperparameter Tuning Step ---
    print("\n>>> 1. TUNING ALS Hyperparameters on the VALIDATION set <<<")

    best_als_params = tune_als_hyperparameters(train_set, validation_set, n_trials=25) 
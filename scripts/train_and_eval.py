import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from utils import prepare_ground_truth, calculate_metrics
from data_prepare import prepare_data, SASRecDataset, SASRecDataModule
from models import recommend_popular_items_and_evaluate, recommend_item_item_and_evaluate, recommend_als_and_evaluate, SASRec


def train_and_eval_SASRec_model(train_set, validation_set, test_set, checkpoint_dir_path='checkpoints/',
                                checkpoint_path=None, n_epochs=10, mode='train',
                                batchsize=256, max_token_len=50, learning_rate=1e-3, hidden_dim=128,
                                num_heads=2, num_layers=2, dropout=0.2, weight_decay=1e-6):
    """
    Train or evaluate a SASRec sequential recommendation model using PyTorch Lightning.

    This function wraps the entire SASRec pipeline:
    - Initializes the SASRecDataModule (handles dataset preprocessing and dataloaders).
    - Builds the SASRec Transformer-based model.
    - Configures training callbacks (checkpointing, early stopping, LR monitoring).
    - Runs either training (`mode='train'`) or evaluation on the test set (`mode='test'`).

    Args
    ----------
    train_set : pd.DataFrame
        Training interactions dataset .
    validation_set : pd.DataFrame
        Validation dataset with the same structure as `train_set`.
    test_set : pd.DataFrame
        Test dataset with the same structure as `train_set`.
    checkpoint_dir_path : str, optional (default='checkpoints/')
        Directory to save model checkpoints.
    checkpoint_path : str or None, optional (default=None)
        Path to a checkpoint file for resuming training or loading a pretrained model for testing.
    n_epochs : int, optional (default=10)
        Number of training epochs.
    mode : {'train', 'test'}, optional (default='train')
        - `'train'`: trains the model on the training/validation data.
        - `'test'`: evaluates the model on the test set using a checkpoint.
    batchsize : int, optional (default=256)
        Batch size for training and evaluation.
    max_token_len : int, optional (default=50)
        Maximum sequence length per user (recent interactions kept).
    learning_rate : float, optional (default=1e-3)
        Learning rate for the AdamW optimizer.
    hidden_dim : int, optional (default=128)
        Dimensionality of item and positional embeddings.
    num_heads : int, optional (default=2)
        Number of attention heads in each Transformer encoder layer.
    num_layers : int, optional (default=2)
        Number of Transformer encoder layers.
    dropout : float, optional (default=0.2)
        Dropout probability applied in embeddings and Transformer layers.
    weight_decay : float, optional (default=1e-6)
        Weight decay regularization coefficient for AdamW.
    """
    # --- 1. Initialize DataModule ---
    print("Initializing DataModule...")
    datamodule = SASRecDataModule(
        train_df=train_set,
        val_df=validation_set,
        test_df=test_set,
        batch_size=batchsize,
        max_len=max_token_len
    )
    datamodule.setup()

    # --- 2. Initialize Model ---
    print("Initializing SASRec model...")
    model = SASRec(
        vocab_size=datamodule.vocab_size,
        max_len=max_token_len,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    # --- 3. Configure Training Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir_path,
        filename="sasrec-{epoch:02d}-{val_hitrate@10:.4f}",
        save_top_k=1,
        verbose=True,
        monitor="val_hitrate@10",
        mode="max"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_hitrate@10",   # stop if ranking metric stagnates
        patience=5,
        mode="max"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = TensorBoardLogger("lightning_logs", name="sasrec")

    # --- 4. Initialize Trainer ---
    print("Initializing PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        max_epochs=n_epochs,
        accelerator='auto',
        devices=1,
        gradient_clip_val=1.0,  # helps with exploding gradients
    )

    if mode == 'train' :
        # --- 5. Start Training ---
        print(f"Starting training for up to {n_epochs} epochs...")
        trainer.fit(model, datamodule,
                    ckpt_path=checkpoint_path
                    )

    elif mode == 'test':
        # --- 6. Test on best checkpoint ---
        print("Evaluating on test set...")
        trainer.test(model, datamodule,
                     ckpt_path=checkpoint_path
                    )

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # --- Configuration ---
    BATCH_SIZE = 256
    MAX_TOKEN_LEN = 50     # 50–100 is standard
    LEARNING_RATE = 1e-3
    HIDDEN_DIM = 128
    NUM_HEADS = 2
    NUM_LAYERS = 2
    DROPOUT = 0.2
    WEIGHT_DECAY = 1e-6
    N_EPOCHS = 50
    CHECKPOINT_SAVE_PATH = 'checkpoints/'
    CHECKPOINT_LOAD_PATH = None  # or specify a path to a checkpoint file
    MODE = 'train'  # 'train' or 'test'
    
    train_set, validation_set, test_set = prepare_data(data_folder='data/')
    if train_set is not None:
        results = {}
        full_train_set = pd.concat([train_set, validation_set])
        
        # Evaluate classical models
        print("\n>>> Running evaluations on the VALIDATION set <<<")
        results['Popularity (Validation)'] = recommend_popular_items_and_evaluate(train_set, validation_set)
        results['Item-Item CF (Validation)'] = recommend_item_item_and_evaluate(train_set, validation_set)
        results['ALS (Validation)'] = recommend_als_and_evaluate(train_set, validation_set)
        
        print("\n>>> Running final evaluations on the TEST set <<<")
        results['Popularity (Test)'] = recommend_popular_items_and_evaluate(full_train_set, test_set)
        results['Item-Item CF (Test)'] = recommend_item_item_and_evaluate(full_train_set, test_set)
        results['ALS (Test)'] = recommend_als_and_evaluate(full_train_set, test_set)
        
        print("\n--- Final Evaluation Results ---")
        results_df = pd.DataFrame.from_dict(results, orient='index')
        print(results_df)
        print("--------------------------------")
        
        # Train and evaluate SASRec model
        print("\n>>> Training and evaluating SASRec model <<<")
        train_and_eval_SASRec_model(train_set, validation_set, test_set, n_epochs=10, mode='train')
        
        print("\n>>> Evaluating trained SASRec model on TEST set <<<")
        train_and_eval_SASRec_model(train_set, validation_set, test_set, mode='test')
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import implicit
from utils import prepare_ground_truth, calculate_metrics

def recommend_popular_items_and_evaluate(train_df, test_df, k=10, prepare_ground_truth=None, calculate_metrics=None):
    """
    Trains a non-personalized Popularity model and evaluates its performance.

    This model recommends the top-k most frequently transacted items from the training
    set to every user. It serves as a simple but strong baseline.

    Args:
        train_df (pd.DataFrame): The training dataset.
        test_df (pd.DataFrame): The test dataset for evaluation.
        k (int): The number of items to recommend.
        prepare_ground_truth (function): A function to process the test_df into a ground truth dict.
        calculate_metrics (function): A function to compute ranking metrics.

    Returns:
        dict: A dictionary containing the calculated evaluation metrics (e.g., precision, recall).
    """
    print(f"\n--- Evaluating Popularity Model (Top {k} items) ---")
    
    # 1. "Train" the model by finding the most popular items based on transactions
    purchase_counts = train_df[train_df['event'] == 'transaction']['itemid'].value_counts()
    popular_items = purchase_counts.head(k).index.tolist()
    print(f"Top {k} popular items identified from training data.")

    # 2. Evaluate the model
    ground_truth = prepare_ground_truth(test_df)
    # Every user receives the same list of popular items
    recommendations = {user_id: popular_items for user_id in ground_truth.keys()}
    
    metrics = calculate_metrics(recommendations, ground_truth, k)
    print("Evaluation complete.")
    return metrics

def recommend_item_item_and_evaluate(train_df, test_df, k=10, min_item_interactions=5, min_user_interactions=5, prepare_ground_truth=None, calculate_metrics=None):
    """
    Trains an Item-Item Collaborative Filtering model and evaluates its performance.

    This model recommends items that are similar to items a user has interacted
    with in the past, based on co-occurrence patterns in the training data.

    Args:
        train_df (pd.DataFrame): The training dataset.
        test_df (pd.DataFrame): The test dataset for evaluation.
        k (int): The number of items to recommend.
        min_item_interactions (int): Minimum number of interactions for an item to be kept.
        min_user_interactions (int): Minimum number of interactions for a user to be kept.
        prepare_ground_truth (function): A function to process the test_df into a ground truth dict.
        calculate_metrics (function): A function to compute ranking metrics.

    Returns:
        dict: A dictionary containing the calculated evaluation metrics.
    """
    print(f"\n--- Evaluating Item-Item CF Model (Top {k} items) ---")
    
    # 1. Filter out infrequent users and items to reduce noise and computation
    item_counts = train_df['itemid'].value_counts()
    user_counts = train_df['visitorid'].value_counts()
    items_to_keep = item_counts[item_counts >= min_item_interactions].index
    users_to_keep = user_counts[user_counts >= min_user_interactions].index
    filtered_df = train_df[(train_df['itemid'].isin(items_to_keep)) & (train_df['visitorid'].isin(users_to_keep))].copy()
    print(f"Filtered training data from {len(train_df)} to {len(filtered_df)} records.")

    # 2. Create user-item interaction matrix and vocabulary mappings
    user_map = {uid: i for i, uid in enumerate(filtered_df['visitorid'].unique())}
    item_map = {iid: i for i, iid in enumerate(filtered_df['itemid'].unique())}
    inverse_item_map = {i: iid for iid, i in item_map.items()}
    user_indices = filtered_df['visitorid'].map(user_map)
    item_indices = filtered_df['itemid'].map(item_map)
    user_item_matrix = csr_matrix((np.ones(len(filtered_df)), (user_indices, item_indices)))

    # 3. Calculate the cosine similarity matrix between all items
    print("Calculating item similarity matrix...")
    item_similarity_matrix = cosine_similarity(user_item_matrix.T, dense_output=False)
    print("Similarity matrix calculated.")

    # 4. Generate recommendations and evaluate
    ground_truth = prepare_ground_truth(test_df)
    recommendations = {}
    print("Generating recommendations for users in test set...")
    test_users = [u for u in ground_truth.keys() if u in user_map]
    
    for user_id in test_users:
        user_index = user_map[user_id]
        user_interactions_indices = user_item_matrix[user_index].indices
        
        if len(user_interactions_indices) > 0:
            # Aggregate scores from items the user has interacted with
            all_scores = np.asarray(item_similarity_matrix[user_interactions_indices].sum(axis=0)).flatten()
            # Remove already interacted items from recommendations
            all_scores[user_interactions_indices] = -1
            top_indices = np.argsort(all_scores)[::-1][:k]
            recs = [inverse_item_map[idx] for idx in top_indices if idx in inverse_item_map]
            recommendations[user_id] = recs
            
    metrics = calculate_metrics(recommendations, ground_truth, k)
    print("Evaluation complete.")
    return metrics

def recommend_als_and_evaluate(train_df, test_df, k=10, min_item_interactions=5, min_user_interactions=5, 
                               factors=25, regularization=0.02, iterations=48, prepare_ground_truth=None, calculate_metrics=None):
    """
    Trains an Alternating Least Squares (ALS) model and evaluates its performance.

    This model uses matrix factorization to learn latent embeddings for users and
    items from implicit feedback data. Default hyperparameters are set from a
    previous Optuna tuning process.

    Args:
        train_df (pd.DataFrame): The training dataset.
        test_df (pd.DataFrame): The test dataset for evaluation.
        k (int): The number of items to recommend.
        min_item_interactions (int): Minimum number of interactions for an item to be kept.
        min_user_interactions (int): Minimum number of interactions for a user to be kept.
        factors (int): The number of latent factors to compute.
        regularization (float): The regularization factor.
        iterations (int): The number of ALS iterations to run.
        prepare_ground_truth (function): A function to process the test_df into a ground truth dict.
        calculate_metrics (function): A function to compute ranking metrics.

    Returns:
        dict: A dictionary containing the calculated evaluation metrics.
    """
    print(f"\n--- Evaluating ALS Model (Top {k} items) ---")
    
    # 1. Filter data
    item_counts = train_df['itemid'].value_counts()
    user_counts = train_df['visitorid'].value_counts()
    items_to_keep = item_counts[item_counts >= min_item_interactions].index
    users_to_keep = user_counts[user_counts >= min_user_interactions].index
    filtered_df = train_df[(train_df['itemid'].isin(items_to_keep)) & (train_df['visitorid'].isin(users_to_keep))].copy()
    print(f"Filtered training data from {len(train_df)} to {len(filtered_df)} records.")

    # 2. Create mappings and confidence matrix
    user_map = {uid: i for i, uid in enumerate(filtered_df['visitorid'].unique())}
    item_map = {iid: i for i, iid in enumerate(filtered_df['itemid'].unique())}
    inverse_item_map = {i: iid for iid, i in item_map.items()}
    user_indices = filtered_df['visitorid'].map(user_map).astype(np.int32)
    item_indices = filtered_df['itemid'].map(item_map).astype(np.int32)
    
    event_weights = {'view': 1, 'addtocart': 3, 'transaction': 5}
    confidence = filtered_df['event'].map(event_weights).astype(np.float32)
    user_item_matrix = csr_matrix((confidence, (user_indices, item_indices)))

    # 3. Train the ALS model
    print("Training ALS model...")
    als_model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations)
    als_model.fit(user_item_matrix)
    print("ALS model trained.")

    # 4. Generate recommendations and evaluate
    ground_truth = prepare_ground_truth(test_df)
    recommendations = {}
    print("Generating recommendations for users in test set...")
    test_users_indices = [user_map[u] for u in ground_truth.keys() if u in user_map]
    
    if test_users_indices:
        user_item_matrix_for_recs = user_item_matrix[test_users_indices]
        ids, _ = als_model.recommend(test_users_indices, user_item_matrix_for_recs, N=k)
        
        for i, user_index in enumerate(test_users_indices):
            original_user_id = list(user_map.keys())[list(user_map.values()).index(user_index)]
            recs = [inverse_item_map[item_idx] for item_idx in ids[i] if item_idx in inverse_item_map]
            recommendations[original_user_id] = recs
            
    metrics = calculate_metrics(recommendations, ground_truth, k)
    print("Evaluation complete.")
    return metrics

class SASRec(pl.LightningModule):
    """
    A PyTorch Lightning implementation of the SASRec model for sequential recommendation.

    SASRec (Self-Attentive Sequential Recommendation) uses a Transformer-based
    architecture to capture the sequential patterns in a user's interaction history
    to predict the next item they are likely to interact with.

    Attributes:
        save_hyperparameters: Automatically saves all constructor arguments as hyperparameters.
        item_embedding (nn.Embedding): Embedding layer for item IDs.
        positional_embedding (nn.Embedding): Embedding layer to encode the position of items in a sequence.
        transformer_encoder (nn.TransformerEncoder): The core self-attention module.
        fc (nn.Linear): Final fully connected layer to produce logits over the item vocabulary.
        loss_fn (nn.CrossEntropyLoss): The loss function used for training.
    """
    def __init__(self, vocab_size, max_len, hidden_dim, num_heads, num_layers,
                 dropout=0.2, learning_rate=1e-3, weight_decay=1e-6, warmup_steps=2000, max_steps=100000):
        """
        Initializes the SASRec model layers and hyperparameters.

        Args:
            vocab_size (int): The total number of unique items in the dataset (+1 for padding).
            max_len (int): The maximum length of the input sequences.
            hidden_dim (int): The dimensionality of the item and positional embeddings.
            num_heads (int): The number of attention heads in the Transformer encoder.
            num_layers (int): The number of layers in the Transformer encoder.
            dropout (float): The dropout rate to be applied.
            learning_rate (float): The learning rate for the optimizer.
            weight_decay (float): The weight decay (L2 penalty) for the optimizer.
            warmup_steps (int): The number of linear warmup steps for the learning rate scheduler.
            max_steps (int): The total number of training steps for the learning rate scheduler's decay phase.
        """
        super().__init__()
        # This saves all hyperparameters to self.hparams, making them accessible later
        self.save_hyperparameters()

        # Embedding layers
        self.item_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.positional_embedding = nn.Embedding(max_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Loss function, ignoring the padding token
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        
        # Lists to store outputs from validation and test steps
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): A batch of input sequences of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The output logits of shape (batch_size, seq_len, vocab_size).
        """
        seq_len = x.size(1)
        # Create positional indices (0, 1, 2, ..., seq_len-1)
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0)

        # Create a causal mask to ensure the model doesn't look ahead in the sequence
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=self.device)

        # Combine item and positional embeddings
        x = self.item_embedding(x) + self.positional_embedding(positions)
        x = self.dropout(x)
        
        # Pass through the Transformer encoder
        x = self.transformer_encoder(x, mask=causal_mask)
        
        # Get final logits
        logits = self.fc(x)
        return logits

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.

        Args:
            batch (tuple): A tuple containing input sequences and target items.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The calculated loss for the batch.
        """
        inputs, targets = batch
        logits = self.forward(inputs)

        # We only care about the prediction for the very last item in the input sequence
        last_logits = logits[:, -1, :]
        
        # Calculate loss against the single target item
        loss = self.loss_fn(last_logits, targets.squeeze())
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.
        Calculates loss and stores predictions for metric computation at the end of the epoch.
        """
        inputs, targets = batch
        logits = self.forward(inputs)
        last_item_logits = logits[:, -1, :]
        loss = self.loss_fn(last_item_logits, targets.squeeze())
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)

        # Get top-10 predictions for metric calculation
        top_k_preds = torch.topk(last_item_logits, 10, dim=-1).indices
        self.validation_step_outputs.append({'preds': top_k_preds, 'targets': targets})
        return loss

    def on_validation_epoch_end(self):
        """
        Calculates and logs ranking metrics at the end of the validation epoch.
        """
        if not self.validation_step_outputs: return

        # Concatenate all predictions and targets from the epoch
        preds = torch.cat([x['preds'] for x in self.validation_step_outputs], dim=0)
        targets = torch.cat([x['targets'] for x in self.validation_step_outputs], dim=0)

        k = preds.size(1)
        # Check if the target is in the top-k predictions for each example
        hits_tensor = (preds == targets).any(dim=1)
        num_hits = hits_tensor.sum().item()
        num_targets = len(targets)

        if num_targets > 0:
            hit_rate = num_hits / num_targets
            recall = hit_rate  # For next-item prediction, recall@k is the same as hit_rate@k
            precision = num_hits / (k * num_targets)
        else:
            hit_rate, recall, precision = 0.0, 0.0, 0.0

        self.log('val_hitrate@10', hit_rate, prog_bar=True)
        self.log('val_precision@10', precision, prog_bar=True)
        self.log('val_recall@10', recall, prog_bar=True)

        self.validation_step_outputs.clear() # Free up memory

    def test_step(self, batch, batch_idx):
        """
        Performs a single test step.
        Mirrors the logic of the validation_step.
        """
        inputs, targets = batch
        logits = self.forward(inputs)
        last_item_logits = logits[:, -1, :]
        loss = self.loss_fn(last_item_logits, targets.squeeze())
        self.log('test_loss', loss, prog_bar=True)

        top_k_preds = torch.topk(last_item_logits, 10, dim=-1).indices
        self.test_step_outputs.append({'preds': top_k_preds, 'targets': targets})
        return loss

    def on_test_epoch_end(self):
        """
        Calculates and logs ranking metrics at the end of the test epoch.
        """
        if not self.test_step_outputs: return

        preds = torch.cat([x['preds'] for x in self.test_step_outputs], dim=0)
        targets = torch.cat([x['targets'] for x in self.test_step_outputs], dim=0)

        k = preds.size(1)
        hits_tensor = (preds == targets).any(dim=1)
        num_hits = hits_tensor.sum().item()
        num_targets = len(targets)

        if num_targets > 0:
            hit_rate = num_hits / num_targets
            recall = hit_rate
            precision = num_hits / (k * num_targets)
        else:
            hit_rate, recall, precision = 0.0, 0.0, 0.0

        self.log('test_hitrate@10', hit_rate, prog_bar=True)
        self.log('test_precision@10', precision, prog_bar=True)
        self.log('test_recall@10', recall, prog_bar=True)

        self.test_step_outputs.clear() # Free up memory

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.
        
        Uses AdamW optimizer and a linear warmup followed by a cosine decay schedule,
        which is a standard practice for training Transformer models.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Learning rate scheduler: linear warmup and cosine decay
        def lr_lambda(current_step: int):
            warmup_steps = self.hparams.warmup_steps
            max_steps = self.hparams.max_steps
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update the scheduler at every training step
                "frequency": 1
            }
        }

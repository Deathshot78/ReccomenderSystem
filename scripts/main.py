import pandas as pd
import numpy as np
import torch
import datetime
from models import SASRec
from utils import prepare_ground_truth, calculate_metrics, load_item_properties, load_category_tree, get_popular_items, show_user_recommendations
from data_prepare import prepare_data, SASRecDataset, SASRecDataModule

def main(checkpoint_path="checkpoints/sasrec-epoch=06-val_hitrate@10=0.3629.ckpt", data_folder="data/"):
    """
    Main function to run the inference and qualitative analysis pipeline.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model from checkpoint...")
    best_model = SASRec.load_from_checkpoint(checkpoint_path)
    best_model.to(device)

    print("Preparing data...")
    train_set, validation_set, test_set = prepare_data(data_folder=data_folder)
    
    datamodule = SASRecDataModule(train_set, validation_set, test_set)
    datamodule.setup()
    
    item_category_map = load_item_properties(data_folder=data_folder)
    category_parent_map = load_category_tree(data_folder=data_folder)
    
    print("\nCalculating popular items for cold-start users...")
    popular_items_list = get_popular_items(train_set, k=10)

    users_in_train_history = set(datamodule.user_history.keys())
    users_in_test_set = set(datamodule.test_df['visitorid'].unique())
    valid_example_users = list(users_in_train_history.intersection(users_in_test_set))

    print(f"\nFound {len(valid_example_users)} users for qualitative analysis.")
    
    for user_id in valid_example_users[:3]:
        show_user_recommendations(user_id, best_model, datamodule, popular_items_list, item_category_map, category_parent_map)
        
    new_user_id = -999
    show_user_recommendations(new_user_id, best_model, datamodule, popular_items_list, item_category_map, category_parent_map)

if __name__ == "__main__":
    main()
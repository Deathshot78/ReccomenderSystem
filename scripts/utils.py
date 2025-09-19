import pandas as pd
import numpy as np
import torch
import datetime

def calculate_metrics(recommendations_dict, ground_truth_dict, k):
    """
    Calculates Precision@k, Recall@k, and HitRate@k.

    args:
    ----------
    recommendations_dict : {user_id: [recommended_item_ids]}
    ground_truth_dict : {user_id: set of ground truth item_ids}
    k : int

    Returns
    -------
    dict with mean precision, recall, and hit rate
    """
    all_precisions, all_recalls, all_hits = [], [], []

    for user_id, true_items in ground_truth_dict.items():
        recs = recommendations_dict.get(user_id, [])[:k]
        if not true_items:
            continue
        hits = len(set(recs) & true_items)

        precision = hits / k if k > 0 else 0
        recall = hits / len(true_items)
        hit_rate = 1.0 if hits > 0 else 0.0

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_hits.append(hit_rate)

    if not all_precisions:
        return {"mean_precision@k": 0, "mean_recall@k": 0, "mean_hitrate@k": 0}

    return {
        "mean_precision@k": np.mean(all_precisions),
        "mean_recall@k": np.mean(all_recalls),
        "mean_hitrate@k": np.mean(all_hits)
    }

def prepare_ground_truth(df, mode="purchase", event_weights=None):
    """
    Prepares ground truth dictionaries for evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Test dataframe containing at least ['visitorid', 'itemid', 'event'].
    mode : str, default="purchase"
        - "purchase" : Only use transactions as ground truth.
        - "all"      : Use all events. Optionally weight them.
    event_weights : dict, optional
        Example: {"view": 1, "addtocart": 3, "transaction": 5}.
        Used only if mode == "all".

    Returns
    -------
    dict : {user_id: set of item_ids}
    """
    if mode == "purchase":
        df_filtered = df[df["event"] == "transaction"]
        ground_truth = df_filtered.groupby("visitorid")["itemid"].apply(set).to_dict()

    elif mode == "all":
        if event_weights is None:
            # Default: treat all events equally
            ground_truth = df.groupby("visitorid")["itemid"].apply(set).to_dict()
        else:
            # Weighted ground truth (for more advanced eval)
            ground_truth = {}
            for uid, user_df in df.groupby("visitorid"):
                weighted_items = []
                for _, row in user_df.iterrows():
                    weight = event_weights.get(row["event"], 1)
                    weighted_items.extend([row["itemid"]] * weight)
                ground_truth[uid] = set(weighted_items)
    else:
        raise ValueError("mode must be 'purchase' or 'all'")

    return ground_truth

def load_item_properties(data_folder='data/'):
    """
    Loads item properties and creates a mapping from item ID to its category ID.
    Handles both a single properties file or two split parts.
    
    Args:
        data_folder (str): The path to the folder containing item property files.

    Returns:
        dict: A dictionary mapping {itemid: categoryid}.
    """
    print("Loading item properties...")
    try:
        # First, try to load the two separate parts and combine them.
        props_df_part1 = pd.read_csv(data_folder + 'item_properties_part1.csv')
        props_df_part2 = pd.read_csv(data_folder + 'item_properties_part2.csv')
        props_df = pd.concat([props_df_part1, props_df_part2], ignore_index=True)
        print("Successfully loaded and combined item_properties_part1.csv and item_properties_part2.csv.")

    except FileNotFoundError:
        try:
            # If the parts are not found, try to load a single combined file.
            props_df = pd.read_csv(data_folder + 'item_properties.csv')
            print("Successfully loaded a single item_properties.csv.")
        except FileNotFoundError:
            print(f"Warning: No item properties files found. Cannot display category information.")
            return {}

    category_df = props_df[props_df['property'] == 'categoryid'].copy()
    category_df['value'] = pd.to_numeric(category_df['value'], errors='coerce').astype('Int64')
    item_to_category_map = category_df.set_index('itemid')['value'].to_dict()
    print("Item to category mapping created successfully.")
    return item_to_category_map

def load_category_tree(data_folder='data/'):
    """
    Loads the category tree to map categories to their parent categories.

    Args:
        data_folder (str): The path to the folder containing category_tree.csv.

    Returns:
        dict: A dictionary mapping {categoryid: parentid}.
    """
    print("Loading category tree...")
    try:
        tree_df = pd.read_csv(data_folder + 'category_tree.csv')
        category_parent_map = tree_df.set_index('categoryid')['parentid'].to_dict()
        print("Category tree loaded successfully.")
        return category_parent_map
    except FileNotFoundError:
        print("Warning: 'category_tree.csv' not found. Cannot display parent category information.")
        return {}

def get_popular_items(train_df, k=10):
    """
    Calculates the top-k most popular items based on transaction count.
    """
    purchase_counts = train_df[train_df['event'] == 'transaction']['itemid'].value_counts()
    return purchase_counts.head(k).index.tolist()

def show_user_recommendations(visitor_id, model, datamodule, popular_items, item_category_map, category_parent_map, k=10):
    """
    Displays recommendations for a user, including category and parent category information.
    """
    print(f"\n--- Recommendations for Visitor ID: {visitor_id} ---")
    model.eval()

    def format_item_with_category(item_id):
        category_id = item_category_map.get(item_id, 'N/A')
        parent_id = category_parent_map.get(category_id, 'N/A') if category_id != 'N/A' else 'N/A'
        return f"Item: {item_id} (Category: {category_id}, Parent: {parent_id})"

    user_history_ids = datamodule.user_history.get(visitor_id)

    if user_history_ids is None:
        print(f"User {visitor_id} not found in training history. Providing popularity-based recommendations.")
        print(f"\nTop {k} Popular Items (Fallback):")
        recs_with_cats = [format_item_with_category(item_id) for item_id in popular_items]
        print(recs_with_cats)
        print("-------------------------------------------------")
        return

    history_with_cats = [format_item_with_category(item_id) for item_id in user_history_ids]
    print(f"User's Historical Interactions:")
    print(history_with_cats)

    history_indices = [datamodule.item_map[i] for i in user_history_ids if i in datamodule.item_map]
    if not history_indices:
        print("None of the user's historical items are in the model's vocabulary.")
        return

    max_len = datamodule.max_len
    input_seq = history_indices[-max_len:]
    padded_input = np.zeros(max_len, dtype=np.int64)
    padded_input[-len(input_seq):] = input_seq
    
    input_tensor = torch.LongTensor(np.array([padded_input]))
    input_tensor = input_tensor.to(model.device)

    with torch.no_grad():
        logits = model(input_tensor)
        last_item_logits = logits[0, -1, :]
        top_indices = torch.topk(last_item_logits, k).indices.tolist()

    recommended_item_ids = [datamodule.inverse_item_map[idx] for idx in top_indices if idx in datamodule.inverse_item_map]

    print(f"\nTop {k} Recommended Items:")
    recs_with_cats = [format_item_with_category(item_id) for item_id in recommended_item_ids]
    print(recs_with_cats)
    print("-------------------------------------------------")

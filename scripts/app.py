import gradio as gr
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from models import SASRec
from data_prepare import SASRecDataset, SASRecDataModule, prepare_data
from utils import load_item_properties, load_category_tree, get_popular_items

# --- Global variables to hold loaded artifacts ---
# This prevents reloading the model and data on every prediction.
MODEL = None
DATAMODULE = None
ITEM_CATEGORY_MAP = None
CATEGORY_PARENT_MAP = None
POPULAR_ITEMS = None

# --- Data Loading and Preparation Functions ---

def load_artifacts():
    """
    Loads all necessary artifacts (model, data, mappings) into global variables.
    This function is called only once when the app starts.
    """
    global MODEL, DATAMODULE, ITEM_CATEGORY_MAP, CATEGORY_PARENT_MAP, POPULAR_ITEMS
    
    print("--- Loading all artifacts for the Gradio app ---")
    
    # HF-FRIENDLY: Path is relative, assuming the checkpoint is in the root of the Space repo.
    CHECKPOINT_PATH = "checkpoints/sasrec-epoch=06-val_hitrate@10=0.3629.ckpt"
    DATA_FOLDER = "data/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from checkpoint: {CHECKPOINT_PATH}...")
    MODEL = SASRec.load_from_checkpoint(CHECKPOINT_PATH)
    MODEL.to(device)
    MODEL.eval()

    print("Preparing data...")
    train_set, validation_set, test_set = prepare_data(data_folder=DATA_FOLDER)
    
    DATAMODULE = SASRecDataModule(train_set, validation_set, test_set)
    DATAMODULE.setup()
    
    print("Loading item and category maps...")
    ITEM_CATEGORY_MAP = load_item_properties(data_folder=DATA_FOLDER)
    CATEGORY_PARENT_MAP = load_category_tree(data_folder=DATA_FOLDER)
    
    print("Calculating popular items for cold-start users...")
    POPULAR_ITEMS = get_popular_items(train_set, k=10)
    
    print("--- Artifacts loaded successfully. Ready to serve recommendations. ---")

def get_recommendations(visitor_id_str):
    """
    The main prediction function for the Gradio interface.
    
    Takes a visitor ID string, gets recommendations, and formats them for display.
    """
    try:
        visitor_id = int(visitor_id_str)
    except (ValueError, TypeError):
        return pd.DataFrame(), pd.DataFrame(), "Please enter a valid numerical Visitor ID."

    user_history_ids = DATAMODULE.user_history.get(visitor_id)

    def format_to_df(item_list):
        data = []
        for rank, item_id in enumerate(item_list, 1):
            category_id = ITEM_CATEGORY_MAP.get(item_id, 'N/A')
            parent_id = CATEGORY_PARENT_MAP.get(category_id, 'N/A') if pd.notna(category_id) else 'N/A'
            data.append([rank, item_id, category_id, parent_id])
        return pd.DataFrame(data, columns=['Rank', 'Item ID', 'Category ID', 'Parent ID'])

    # --- Cold-Start User (Fallback to Popularity) ---
    if user_history_ids is None:
        history_df = pd.DataFrame(columns=['Rank', 'Item ID', 'Category ID', 'Parent ID'])
        recs_df = format_to_df(POPULAR_ITEMS)
        message = f"User {visitor_id} is new. Showing Top 10 popular items as a fallback."
        return history_df, recs_df, message

    # --- Existing User (Use SASRec Model) ---
    history_df = format_to_df(user_history_ids)
    
    history_indices = [DATAMODULE.item_map[i] for i in user_history_ids if i in DATAMODULE.item_map]
    
    if not history_indices:
        message = "None of this user's historical items are in the model's vocabulary."
        return history_df, pd.DataFrame(), message

    max_len = DATAMODULE.max_len
    input_seq = history_indices[-max_len:]
    padded_input = np.zeros(max_len, dtype=np.int64)
    padded_input[-len(input_seq):] = input_seq
    
    input_tensor = torch.LongTensor(np.array([padded_input]))
    input_tensor = input_tensor.to(MODEL.device)

    with torch.no_grad():
        logits = MODEL(input_tensor)
        last_item_logits = logits[0, -1, :]
        top_indices = torch.topk(last_item_logits, 10).indices.tolist()

    recommended_item_ids = [DATAMODULE.inverse_item_map[idx] for idx in top_indices if idx in DATAMODULE.inverse_item_map]
    recs_df = format_to_df(recommended_item_ids)
    message = f"Showing personalized SASRec recommendations for user {visitor_id}."
    
    return history_df, recs_df, message

# --- Main Execution Block ---
if __name__ == "__main__":
    # Load all artifacts once at startup
    load_artifacts()

    # Find some valid example users to show in the UI
    users_in_train_history = set(DATAMODULE.user_history.keys())
    users_in_test_set = set(DATAMODULE.test_df['visitorid'].unique())
    valid_example_users = list(users_in_train_history.intersection(users_in_test_set))
    
    # Convert numpy types to standard Python int for Gradio compatibility
    example_list = [int(u) for u in valid_example_users[:4]] + [-999]

    # Create and launch the Gradio interface
    with gr.Blocks(theme=gr.themes.Soft(), title="SASRec Recommender") as iface:
        gr.Markdown(
            """
            # SASRec Sequential Recommender System
            An interactive demo of a state-of-the-art recommender system trained on the RetailRocket dataset.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                visitor_id_input = gr.Number(
                    label="Enter Visitor ID", 
                    info="Enter a user's numerical ID to get recommendations."
                )
                submit_button = gr.Button("Get Recommendations", variant="primary")
                gr.Examples(
                    examples=example_list,
                    inputs=visitor_id_input,
                    label="Example User IDs (Click to try)"
                )
            
            with gr.Column(scale=3):
                status_message = gr.Textbox(label="Status", interactive=False)
                with gr.Tabs():
                    with gr.TabItem("Top 10 Recommendations"):
                        recs_output = gr.DataFrame(label="Recommended Items")
                    with gr.TabItem("User's Recent History"):
                        history_output = gr.DataFrame(label="Interaction History")
            
        submit_button.click(
            fn=get_recommendations,
            inputs=visitor_id_input,
            outputs=[history_output, recs_output, status_message]
        )

    # For local testing, this creates a shareable link.
    # On Hugging Face Spaces, this is not strictly necessary but doesn't hurt.
    iface.launch(share=True)
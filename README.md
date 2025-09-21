![Recomm](assets/banner.png)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C?logo=pytorch)](https://pytorch.org/)![Made with ML](https://img.shields.io/badge/Made%20with-ML-blueviolet?logo=openai)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# 🚀 End-to-End Sequential Recommender System  

This project implements and evaluates a series of recommender system models, culminating in a state-of-the-art **SASRec (Self-Attentive Sequential Recommendation)** model for Top-N next-item prediction. The system is trained on the [RetailRocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) and includes an interactive web demo built with Gradio.  

![Gradio app](assets/gradio.png)
You can find the Gradio app [Here](https://huggingface.co/spaces/Deathshot78/RecommenderSystem)

---

## 📑 Table of Contents  

- [📖 Project Overview](#-project-overview)  
- [✨ Key Features](#-key-features)  
- [🧩 Models Implemented](#-models-implemented)  
- [📊 Final Results](#-final-results)  
- [🔍 Qualitative Analysis](#-qualitative-analysis)  
- [🚧 Future Improvements](#-future-improvements)  
- [📂 Project Structure](#-project-structure)  
- [⚙️ Setup and Usage](#️-setup-and-usage)  
- [🛠️ Technologies and Models Used](#️-technologies-and-models-used)  

---

## 📖 Project Overview  

The primary goal of this project is to predict the next item a user is likely to interact with based on their recent session history. This is a common and critical task in e-commerce known as Top-N sequential recommendation.  

The project follows a structured approach:  

1. **Baseline Models**: Simple, non-sequential models to establish a performance baseline.  
2. **Hyperparameter Tuning**: Optuna is used to find the optimal configuration for ALS.  
3. **Advanced Sequential Model**: Implementation of **SASRec** with PyTorch Lightning.  
4. **Evaluation**: Offline evaluation using ranking metrics (Hit Rate, Precision, Recall @ 10).  
5. **Interactive Demo**: A Gradio web app for real-time personalized and cold-start recommendations.  

---

## ✨ Key Features  

- 🔹 **Comprehensive Model Comparison**: From popularity to Transformer-based SASRec.  
- 🔹 **Robust Evaluation**: Time-based data split for realistic performance measurement.  
- 🔹 **Hyperparameter Optimization**: Automated with Optuna for ALS.  
- 🔹 **Deep Learning with Attention**: Full PyTorch Lightning implementation of SASRec.  
- 🔹 **Interactive Web Demo**: Live Gradio app for recommendations.  
- 🔹 **Modular Codebase**: Clean, organized structure.  

---

## 🧩 Models Implemented  

| Model | Methodology | Key Characteristics |
| :--- | :--- | :--- |
| **Popularity** | Non-personalized | Recommends the most frequently purchased items across all users. |
| **Item-Item CF** | Collaborative Filtering | Recommends items similar to a user’s past interactions. |
| **ALS** | Matrix Factorization | Learns latent embeddings from implicit feedback, tuned with Optuna. |
| **SASRec** | Transformer (Self-Attention) | Sequential model capturing contextual user-item interactions. |

---

## 📊 Final Results  

SASRec significantly outperformed all baselines, with a **~4.7x improvement in Hit Rate**.  

| Model | Test Hit Rate@10 | Test Precision@10 | Test Recall@10 |
| :--- | :---: | :---: | :---: |
| Popularity | 0.0651 | 0.0065 | 0.0324 |
| Item-Item CF | 0.0021 | 0.0002 | 0.0011 |
| Tuned ALS | 0.0063 | 0.0006 | 0.0042 |
| **SASRec** | **0.3069** | **0.0307** | **0.3069** |

---

## 🔍 Qualitative Analysis  

The SASRec model not only recommends previously viewed items but also discovers **new, contextually relevant items**.  
For example, for a user browsing **Category 1279**, SASRec suggested new items from the same category — showing strong personalization and discovery.  

---

## 🚧 Future Improvements  

- 📦 **Incorporate Item Features** (e.g., from `item_properties.csv`).  
- 🤖 **Explore Advanced Models**:  
  - BERT4Rec (bidirectional Transformers).  
  - Graph-based recommender systems.  
- 🧪 **Online A/B Testing** for business impact.  
- ⚡ **Scalability Enhancements**: Feature stores, inference servers (Triton), quantization, distillation.  

---

## 📂 Project Structure  

```bash
├── checkpoints/              # Saved PyTorch Lightning checkpoints
├── data/                     # RetailRocket dataset
├── notebooks/                # EDA notebooks
└── scripts/                  
    ├── als_optuna_study.py   # Optuna tuning for ALS
    ├── app.py                # Gradio web demo
    ├── data_prepare.py       # Data loading & preprocessing
    ├── main.py               # Entry point for demo
    ├── models.py             # Model definitions
    ├── train_and_eval.py     # Training & evaluation loop
    └── utils.py              # Helper functions
├── README.md  
└── requirements.txt  
```

---

## ⚙️ Setup and Usage

Follow these steps to set up and run the project locally.

### 1. Prerequisites

- Python 3.10.6+
- An NVIDIA GPU is recommended for training the SASRec model.

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 3. Install all required packages

```bash
pip install -r requirements.txt
```

### 4. Download and Place Data

- Download the [RetailRocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

Then run this script:

```bash
python data_prepare.py
```

### 5. Run the Full Evaluation

To train all models and see the final comparison table, run the main script:

```bash
python train_and_eval.py
```

### 6. Run the main script

```bash
python main.py
```

---

## 🛠️ Technologies and Models Used

This project leverages a range of modern data science and machine learning technologies to build a robust recommender system from the ground up.

### 🏭 Models

- **Popularity Model**: A non-personalized baseline that recommends the most frequently purchased items.
- **Item-Item Collaborative Filtering**: A classical neighborhood-based model that recommends items based on co-occurrence patterns with a user's interaction history.
- **Alternating Least Squares (ALS)**: A powerful matrix factorization technique for implicit feedback, optimized with hyperparameter tuning.
- **SASRec (Self-Attentive Sequential Recommendation)**: A state-of-the-art sequential model based on the Transformer architecture, designed to capture the order and context of user interactions.

### 👩‍💻 Core Technologies & Libraries

- **Python 3.10**: The primary programming language for the project.
- **Pandas & NumPy**: For efficient data manipulation, preprocessing, and numerical operations.
- **Scikit-learn**: Used for calculating item similarity in the collaborative filtering model.
- **Implicit**: For the ALS model
- **PyTorch & PyTorch Lightning**: The deep learning framework used to build, train, and evaluate the SASRec model in a structured and scalable way.
- **Optuna**: A hyperparameter optimization framework used to automatically find the best parameters for the ALS model.
- **Gradio**: A fast and simple framework used to build and deploy the interactive web demo.
- **TensorBoard**: For logging and visualizing model training metrics.

# Movie recommendation system - two tower model

## Introduction

This is my final project for the STAT 542 *Statistical Learning* course I took during my master's degree at the University of Illinois at Urbana-Champaign. It focuses on implementing a newer deep learning model — the Two-Tower model — for recommender systems and comparing it with traditional recommendation methods. The project aims to predict the unknown rating \( R_{ij} \) that user *i* would give to movie *j*.
Note: This was a group project. I was responsible for implementing the Two-Tower model, while my teammates handled the baseline models (e.g., Soft/Hard Impute and SVD).

## Data Source

The dataset used in this project is the [MovieLens 100K](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset). It contains 100,000 movie ratings (1–5 stars) from 943 users on 1,682 movies, along with movie metadata such as genres.

## Repository Structure

```plaintext

TwoTower_MovieLens_Project/
│
├── data/
│   ├── u.data                  # MovieLens ratings data
│   ├── u.item                  # Movie metadata (genres)
│   ├── u.user                  # User metadata
│
├── notebooks/ # Folder contains .ipynb and .pdf versions of the code for easy review.   
│   ├── Final_tt_model.ipynb      # all Two-Tower model codes
│   ├── Final_tt_model.pdf
│
├── reports/
│   ├── Final_Report.pdf        # Final project report
│   ├── Final_Presentation.pdf  # Presentation slides
│
└── README.md        

```

## Tools and technical skills

- **Python**

- **PyTorch** for Two-Tower model implementation
- **Scikit-learn** for preprocessing (Label Encoding)
- **Pandas, NumPy** for data processing
- **Matplotlib, Seabron** for visualization

### Technical Details

## Data Preparation

1. **Train-test split**: 
    - For each user, if the user has < 5 ratings, all ratings go to training.
    - For users with ≥ 5 ratings, randomly split **80\% training** / **20\% test**.

2. **Normalization**:
    - Ratings were **mean-centered per user**, then scaled to [-1, 1] range using each user's max absolute deviation.
    - This ensures that personal rating tendencies are removed, allowing the model to learn relative preferences.

3. **Feature encoding**:
    - User ID and Movie ID: Label encoded.
    - Gender and occupation: Encoded as integers.
    - Age: Normalized via tf.keras Normalization layer.
    - Genres: 19-dim one-hot vector.

## Model Architecture

Implemented a TensorFlow-based **Two-Tower model**:
- **User tower**:
    - User embedding
    - Gender embedding
    - Occupation embedding
    - Age normalization + dense layer
    - User dense layers → final user embedding.

- **Movie tower**:
    - Movie embedding
    - Genre dense layer
    - Movie dense layers → final movie embedding.

- The **cosine similarity** between user and movie embeddings is used to predict the rating.

## Loss Functions

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

→ The model is optimized for **rating prediction** instead of strict top-K ranking.

## Evaluation Metrics

- **RMSE** and **MAE** on test set to measure rating prediction accuracy.
- Visualized **train and test RMSE/MAE** across different hyperparameter settings.
- Benchmarked Two-Tower model against traditional matrix factorization methods:
    - **Singular Value Decomposition (SVD)**
    - **Soft Impute** and **Hard Impute**

## Experiment Design

- Conducted grid search over:
    - Embedding sizes: [16, 32, 64]
    - User dense layer sizes: [8, 16, 32, 64, 128]
    - Movie dense layer sizes: [8, 16, 32, 64, 128]
    - Final embedding dimension: [64, 128]

- Ran **120+ experiments**.
- Identified best configurations for **low test RMSE and MAE**.


## Key Findings

- Achieved best **test RMSE ≈ 0.396**.
- The architecture is sensitive to the final embedding dimension and dense layer sizes.
- Larger models may overfit unless properly tuned.




## Contact Information

For any further questions or collaboration opportunities, please reach out to me at:
- Email: [yguo8395@gmail.com](mailto:yguo8395@gmail.com)
- LinkedIn: [Iris Kuo](https://www.linkedin.com/in/yi-hsuan-kuo-835b00268/)
- GitHub: [Iris Kuo](https://github.com/Iris910531)

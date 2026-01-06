# AutoJudge: Automatic Programming Problem Difficulty Estimation

#### Mishka Singla

#### 24116052

#### ECE (B.Tech. 2nd Year)

## Project Overview

Online competitive programming platforms such as **Codeforces, CodeChef, and Kattis** assign difficulty labels (Easy / Medium / Hard) and numerical difficulty scores to programming problems. These labels are largely based on **human judgment and community feedback**, making the process subjective and time-consuming.

This project, **AutoJudge**, aims to **automatically estimate the difficulty of programming problems using only textual information**, without relying on tags, constraints metadata, or user statistics.

The system performs **two tasks**:

1. **Classification Task**  
   Predict the difficulty class → **Easy / Medium / Hard**

2. **Regression Task**  
   Predict a **numerical difficulty score**

A **Streamlit web application** is provided where users can paste a new problem description and instantly receive both predictions.

---

## Objectives

- Use **only textual content** of programming problems
- Build:
  - a **classification model** for difficulty class
  - a **regression model** for difficulty score
- Understand the limitations of text-based difficulty estimation
- Deploy the solution via a **simple web UI**

---

## Dataset Description

The dataset is taken from this repository. 

https://www.google.com/url?q=https://github.com/AREEG94FAHAD/TaskComplexityEval-24&sa=D&source=editors&ust=1767188820861202&usg=AOvVaw3MOyl1JRoI3VdQ01DE3s6W

Each data sample contains:

- `title`
- `description`
- `input_description`
- `output_description`
- `problem_class` → Easy / Medium / Hard
- `problem_score` → numerical difficulty value

 The dataset is assumed to be **pre-labeled** and is not created or annotated as part of this project.

---

## Text Preprocessing

All textual fields are **combined into a single input string**:

title + description + input_description + output_description


Preprocessing steps:
- Lowercasing text
- Removing extra whitespace
- No aggressive stemming or lemmatization (to preserve programming-specific terms like `dp`, `dfs`, `mod`)

---

## Feature Engineering

To capture both semantic and structural information, a **hybrid feature representation** was used.

### 🔹 TF-IDF Features
- Unigrams and bigrams
- Captures important keywords and phrases
- Forms the core representation of problem text

### 🔹 Hand-Crafted Structural Features
Added to inject domain knowledge:

1. **Log-scaled text length**  
   Longer problems often correspond to higher difficulty

2. **Algorithmic keyword frequency**  
   Keywords such as:

   dp, graph, dfs, bfs, recursion, binary search, greedy

   
3. **Mathematical symbol density**  
Frequency of symbols like `= < > + - * %`, indicating mathematical complexity

All features are concatenated as:

[ TF-IDF | text length | keyword count | math symbol density ]


---

## Classification Models (Difficulty Class)

### Models Tried

| Model | Outcome |
|----|----|
| Logistic Regression | Collapsed to majority class |
| Multinomial Naive Bayes | Weak separation, strong bias |
| **Linear SVM** | Best balanced performance |

---

### Problems Faced During Classification

1. **Severe Class Imbalance**
   - “Hard” problems dominated the dataset
   - Initial models predicted only the majority class

2. **Overlapping Vocabulary**
   - Easy, Medium, and Hard problems share very similar wording
   - Boilerplate input/output formats reduce discriminative power

3. **Subjective Labels**
   - Difficulty categories are not sharply defined even for humans

---

### Final Classification Model

- **Linear Support Vector Machine (LinearSVC)**
- TF-IDF + structural features
- `class_weight="balanced"`

### Classification Results

- Accuracy ≈ **47%**
- Hard problems are identified more reliably
- Easy vs Medium confusion remains high

This performance is **realistic** for a text-only difficulty classification task.

---

## Regression Models (Difficulty Score)

Unlike classification, predicting a **continuous score** is often more stable.

### Models Tried

| Model | MAE | RMSE |
|----|----|----|
| Linear Regression | Weak baseline |
| **Random Forest Regressor** | **1.70** | **2.04** |
| Gradient Boosting Regressor | 1.70 | 2.04 |

---

### Final Regression Model

- **Random Forest Regressor**
- TF-IDF + structural features
- Chosen for:
  - Slightly better robustness
  - Easier interpretability
  - Comparable performance to Gradient Boosting

---

### Regression Evaluation

- **MAE ≈ 1.7**
- **RMSE ≈ 2.0**
- Difficulty score range ≈ 1–10

This corresponds to a **~20% relative error**, which is reasonable given:
- Subjective ground truth
- Use of text-only information

Regression consistently outperformed classification in reliability.

---

## Trained Models and Artifacts

The `models/` directory contains all trained models and preprocessing components required for inference. Each artifact is serialized using `joblib` and loaded at runtime to ensure consistency between training and deployment.

### Contents

| File | Purpose |
|----|----|
| `svm_classifier.pkl` | Predicts difficulty class (Easy / Medium / Hard) |
| `rf_regressor.pkl` | Predicts numerical difficulty score |
| `tfidf_classifier.pkl` | TF-IDF vectorizer for classification |
| `tfidf_regressor.pkl` | TF-IDF vectorizer for regression |
| `feature_scaler.pkl` | Scales handcrafted structural features |

### 🔹 Inference Pipeline

1. Input text is transformed using pre-fitted TF-IDF vectorizers.
2. Structural features are extracted and scaled using the saved scaler.
3. Predictions are generated using:
   - Linear SVM for classification
   - Random Forest Regressor for regression

This design avoids retraining during inference and ensures reproducible predictions.


## Streamlit Web Interface

The project includes a **Streamlit-based web interface** that allows users to interact with the trained models without requiring any technical setup or retraining.

### 🔹 Interface Features

The web application provides:
- Text input fields for:
  - **Problem Description**
  - **Input Description**
  - **Output Description**
- A **Predict** button to trigger inference
- Real-time display of:
  - **Predicted Difficulty Class** (Easy / Medium / Hard)
  - **Predicted Difficulty Score** (numerical value)

---

### 🔹 How It Works

1. The user pastes the problem text into the input fields.
2. The text fields are combined and preprocessed.
3. The same **TF-IDF vectorizers and feature engineering pipeline** used during training are applied.
4. The trained models are loaded from disk:
   - Linear SVM for difficulty classification
   - Random Forest Regressor for difficulty score prediction
5. Predictions are displayed instantly on the interface.

The application performs **inference only** and does not retrain models at runtime.

---

### 🔹 Technology Used

- **Streamlit** for building the interactive web UI
- **joblib** for loading serialized models
- **scikit-learn** for model inference
- **NumPy / SciPy** for feature handling

---

### 🔹 Purpose of the Web Interface

The web interface demonstrates the practical applicability of the system by:
- Making the model accessible to non-technical users
- Allowing real-time testing on unseen problems
- Providing a clean and intuitive way to showcase results

This interface serves as a lightweight deployment layer for the trained machine learning pipeline.


### Run the Project Locally

Follow the steps below to run the AutoJudge application on your local machine.

#### 1️⃣ Clone the repository

```bash
git clone https://github.com/mishka-s/AutoJudge.git
cd AutoJudge

```

#### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt

```

#### 3️⃣ Run the Streamlit Application

```bash
streamlit run app.py

```

Paste a problem description, input description, and output description into the interface to obtain the predicted difficulty class and difficulty score.

## Link to the demo video

   [https://drive.google.com/file/d/12rPp42BXvyZm_7rlG5naIhGuxMy_FHuP/view?usp=drive_link](https://drive.google.com/file/d/1vlAEnO3Mn_CQcvebRg4ZBM4Xi3P8ENFd/view?usp=drive_link)


## Repository Structure

```text
acm/
├── app.py
├── README.md
├── requirements.txt
│
├── models/
│   ├── svm_classifier.pkl
│   ├── rf_regressor.pkl
│   ├── tfidf_classifier.pkl
│   ├── tfidf_regressor.pkl
│   └── feature_scaler.pkl
│
├── notebooks/
│   └── jupyter.ipynb
│
└── data/
    └── problems_data.jsonl

```

Large model files are tracked using Git LFS.

## Key Learnings & Insights

- Difficulty classification is **harder than score regression**
- Text alone has limited ability to distinguish Easy vs Medium
- Structural features improve performance marginally
- Model performance plateaus due to inherent task ambiguity
- Deployment requires **feature consistency** between training and inference

---

## Future Improvements

- Parse constraint values explicitly (e.g., `n ≤ 10^5`)
- Hierarchical classification (Easy vs Non-Easy)
- Use transformer-based embeddings
- Incorporate problem tags if available

---

## Conclusion

**AutoJudge** demonstrates a complete end-to-end NLP + ML pipeline for estimating programming problem difficulty using textual data alone. While classification accuracy is inherently limited by subjective labels, regression-based difficulty estimation provides meaningful and stable predictions. The project highlights both the **potential and limitations** of automated difficulty estimation.


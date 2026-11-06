# SC5002 Lab Assignment 2
# Machine Learning: K-Means Clustering and Bank Subscription Prediction

A collection of three hands-on machine learning experiments exploring **unsupervised clustering**, the limitations of distance-based algorithms, and **supervised binary classification with PyTorch**.

Developed as part of **SC5002 AI Fundamentals** at Nanyang Technological University (NTU), Singapore. Generative AI tools, including ChatGPT and Claude, were used to assist with coding and debugging for this assignment.

## Project overview

| Exercise | Problem | Methods | Key learning |
|---|---|---|---|
| 1. Customer segmentation | Group customers by annual income and spending score | K-Means, StandardScaler, Elbow Method | Choosing cluster counts and understanding initialisation sensitivity |
| 2. Circular clustering | Group 30 synthetic points arranged in circular structures | K-Means, visual comparison with ground-truth labels | Why centroid-based clustering can fail on non-convex shapes |
| 3. Bank subscription prediction | Predict whether a customer subscribes to a term deposit | PyTorch feedforward neural network, preprocessing, evaluation, learning-rate ablation | Classification performance, class imbalance, and overfitting |

## Technologies

- **Language:** Python
- **Data processing:** NumPy, Pandas
- **Machine learning:** Scikit-learn
- **Deep learning:** PyTorch
- **Visualisation:** Matplotlib
- **Data source:** UCI Machine Learning Repository (`ucimlrepo`)
- **Environment:** Jupyter Notebook

## Exercise 1 — K-Means customer segmentation

### Objective

Investigate whether K-Means can identify meaningful customer segments from **20 customers**, represented by two numerical attributes: annual income (in thousands of dollars) and spending score.

### Method

1. Construct the 20-customer dataset in the notebook.
2. Standardise both features using `StandardScaler` to prevent differences in feature scale from dominating Euclidean distances.
3. Fit K-Means with **k = 3** and **k = 5** and visualise the resulting clusters and centroids.
4. Compute within-cluster sum of squares (WCSS/inertia) for **k = 1 through 6** and plot the Elbow Method curve.
5. Fix **k = 3** and compare random seeds **1, 42, and 84** to study initialisation sensitivity.

### Takeaways

K-Means provides an interpretable starting point for customer segmentation, but cluster membership depends on the chosen number of clusters and can be sensitive to centroid initialisation. WCSS decreases as the number of clusters increases, so the Elbow Method is a heuristic rather than a guarantee of the optimal k.

## Exercise 2 — Limitations of K-Means on circular data

### Objective

Test whether standard K-Means can recover the true grouping of **30 two-dimensional synthetic observations** with a circular, non-convex structure.

### Method

1. Load the 30 predefined points and their true labels.
2. Fit K-Means with **k = 2** (`random_state=42`, `n_init=50`).
3. Plot the ground-truth labels alongside the predicted clusters and centroids.
4. Compare the geometry of the resulting partitions with the actual groups.

### Takeaways

K-Means partitions space according to proximity to centroids, producing convex Voronoi regions. It therefore struggles to recover circular or otherwise non-convex groupings, even when the correct number of clusters is specified. Alternatives worth exploring include DBSCAN and spectral clustering.

## Exercise 3 — Bank customer subscription prediction

### Problem

Predict whether a customer will subscribe to a term deposit (`yes`/`no`) using information from a Portuguese bank's direct marketing campaigns. A useful classifier could help prioritise outreach, although predictive performance must be assessed separately for subscribers and non-subscribers.

### Dataset

- **Source:** [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing), retrieved through `fetch_ucirepo(id=222)`
- **Observations:** 45,211
- **Original input features:** 16
- **Model input features:** 15 (the `duration` feature is excluded because call duration is not known before the call)
- **Target:** `y` (`yes` = 1; `no` = 0)
- **Class distribution:** 39,922 non-subscribers and 5,289 subscribers

### Data preparation

1. Inspect sample records, feature types, summary statistics, and the target distribution.
2. Remove `duration` to avoid relying on information unavailable at prediction time.
3. Use a **stratified 60% / 20% / 20%** training/validation/test split (`random_state=42`).
4. Standardise numerical columns and one-hot encode categorical columns using `ColumnTransformer`.
5. Fit the preprocessing pipeline **on training data only**, then transform validation and test data with the fitted pipeline.
6. Convert processed features and labels to PyTorch tensors and construct mini-batch data loaders.

The resulting model has **50 encoded input features** in the saved notebook run.

### Neural network architecture

```text
Input: 50 encoded features
    ↓
Linear(50, 32) + ReLU
    ↓
Linear(32, 16) + ReLU
    ↓
Linear(16, 1) → output logit
```

| Setting | Value |
|---|---|
| Framework | PyTorch |
| Model | Feedforward neural network |
| Hidden layers | 32 and 16 neurons, each followed by ReLU |
| Loss | `BCEWithLogitsLoss` |
| Optimiser | Adam |
| Baseline learning rate | 0.001 |
| Batch size | 64 |
| Training epochs | 100 |
| Prediction threshold | 0.5 after sigmoid |

### Model evaluation

The notebook reports the following performance for the final trained model:

| Dataset | Loss | Accuracy | Precision (subscriber) | Recall (subscriber) |
|---|---:|---:|---:|---:|
| Training | 0.2472 | 91.22% | 73.51% | 38.92% |
| Validation | 0.3125 | 89.21% | 57.92% | 28.36% |
| Test | 0.3081 | **89.03%** | **55.98%** | **29.21%** |

**Interpretation:** Accuracy is relatively high, but the model misses many actual subscribers. Because the dataset is imbalanced, accuracy alone is not sufficient to evaluate whether the model is useful for identifying positive cases.

The notebook's case study reports **7,742 correctly classified non-subscribers** and **309 correctly classified subscribers**, with **749 missed subscribers**. These figures highlight the importance of examining the confusion matrix and class-specific recall.

### Overfitting analysis

Training loss generally continues decreasing while validation loss rises later in training. This suggests overfitting and motivates techniques such as early stopping, dropout, regularisation, and better model selection based on validation performance.

### Learning rate ablation study

Three Adam learning rates were compared with the same model architecture and initial weights, each trained for 100 epochs:

| Learning rate | Final validation loss (saved run) |
|---|---:|
| 0.1 | 0.3610 |
| 0.01 | 0.3600 |
| **0.001** | **0.3153** |

Among these three configurations, **0.001 produced the lowest final validation loss** in the saved ablation experiment. This does not establish that it is globally optimal; further tuning and repeated runs would improve confidence.

## Repository structure

```text
machine-learning-clustering-and-classification/
├── README.md
├── machine_learning_lab.ipynb
├── requirements.txt
└── .gitignore
```

The notebook contains the three exercises, explanatory commentary, plotted visualisations, and saved outputs. Its neural-network architecture cell also writes `bank_neural_network.png` when executed.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/machine-learning-clustering-and-classification.git
cd machine-learning-clustering-and-classification
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create and activate a virtual environment

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open `machine_learning_lab.ipynb` and choose **Run All**. The UCI dataset is downloaded automatically by the notebook, so internet access is needed on the first run. Training can take several minutes depending on hardware.

> **Reproducibility note:** The notebook sets NumPy/PyTorch seeds and K-Means random states, but results can still vary across package versions and compute environments. The notebook also contains a `%pip install -q ucimlrepo` cell; the requirements file installs this package in advance.

## Limitations and future improvements

- **Class imbalance:** Try class-weighted loss, resampling, and threshold tuning; evaluate recall, F1-score, and precision–recall curves.
- **Overfitting:** Implement early stopping and dropout, and restore the best validation checkpoint.
- **Model selection:** Compare logistic regression, random forests, and gradient boosting against the neural network.
- **Clustering:** Evaluate silhouette scores and compare K-Means with density-based or spectral clustering on non-convex data.
- **Evaluation protocol:** Avoid using the test set for epoch-by-epoch model selection; reserve it for a final evaluation after choosing the model using validation data.
- **Reproducibility:** Record exact package versions and run experiments across multiple random seeds.

## Key skills demonstrated

**Exploratory data analysis · Feature scaling · One-hot encoding · K-Means · Clustering diagnostics · PyTorch · Neural network training · Binary classification · Confusion matrices · Precision and recall · Overfitting analysis · Hyperparameter ablation**

## Acknowledgements

- Dataset: [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing).
- Coursework: SC5002 AI Fundamentals, Nanyang Technological University, Singapore.
- Generative AI tools, including ChatGPT and Claude, were used to assist with coding, debugging, and refining the written report for the original assignment.

## Author

**Tan Zi Heng**  
BSc Mathematical Sciences, Nanyang Technological University, Singapore

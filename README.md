# 🏆 M5 Forecasting Accuracy Competition

## 📊 Project Overview

This project implements a solution for the [M5 Forecasting Accuracy competition](https://www.kaggle.com/c/m5-forecasting-accuracy) from Kaggle. The goal of this competition is to predict sales data for Walmart stores across different states, departments, and product categories. 

The repository provides a fully containerized, reproducible pipeline that mirrors the structure and logic of the original Jupyter Notebook-based workflow. This allows for easy, consistent setup and execution without manual configuration.

The project is divided into two key notebooks:

🧪 Technical Notebook:
Includes exploratory data analysis (EDA), data preprocessing, model benchmarking, hyperparameter tuning, and final model evaluation.

💼 Business Analysis Notebook:
Offers a comprehensive, business-oriented interpretation of the forecasting results. It provides insight into trends, risks, and actionable strategies from a business decision-making perspective.

This README serves as a technical guide and quickstart manual for setting up and running the pipeline.

## 📋 Prerequisites

Before diving into the detailed steps of setting up and using this project, there are few important prerequisites or requirements that need to be addressed. These prerequisites ensure that your local development environment is ready and capable of efficiently running and supporting the project.

### Forking and Cloning from GitHub
Create a copy of this repository by forking it on GitHub.

Clone the forked repository to your local machine:

```bash
git clone https://github.com/Sandrog112/M5-Forecasting-Accuracy-Competition
```

### Setting Up Development Environment
Ensure you have Python 3.10+ installed on your machine. 

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```
Also make sure to have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed

### 📁 Project structure:

This project has a modular structure, where each folder has a specific duty.

```bash
.
├── data/                  # Dataset files (raw and processed data)
├── logs/                  # Logs generated during runs and experiments
├── models/                # Saved models
├── outputs/               # Saved outputs from training and evaluation
├── notebook/              # Main jupyter notebooks of the project
│   ├── Business-part.ipynb    # Business-oriented analysis and insights
│   └── Technical-part.ipynb   # Technical exploration and modeling
├── src/                   # Source code for data processing, training, and inference
│   ├── data_preprocessing.py # Script for cleaning and preparing data
│   ├── __init__.py            # Package initializer
│   ├── train/                 # Training scripts and Dockerfile
│   │   ├── train.py           # Main training script
│   │   ├── Dockerfile         # Docker configuration for training environment
│   │   └── __init__.py        # Package initializer for train module
│   └── inference/             # Inference scripts and Dockerfile
│       ├── inference.py       # Script for making predictions with the trained model
│       ├── Dockerfile         # Docker configuration for inference environment
│       └── __init__.py        # Package initializer for inference module
├── .gitignore             # List of files and folders ignored by git
├── README.md              # Project overview and documentation
└── requirements.txt       # Python dependencies needed for the project

```

## 📥 Dataset Setup

In compliance with competition rules, the dataset files are not included in this repository.

Before you run the project, you’ll need to manually download the original dataset by following this steps:

1. Go to: [M5 Forecasting Accuracy competition page](https://www.kaggle.com/c/m5-forecasting-accuracy)

2. Take part in competition by pressing late submission and authorizing your account.

3. After the second step you will be able to download all the sufficient datasets.

4. And finally you create a folder named data in the root of cloned repository and place all csv files into this folder.

Final data folder should look like this:

```bash
├── data/
│   ├── sales_train_validation.csv
│   ├── calendar.csv
│   ├── sell_prices.csv
│   ├── sales_train_evaluation.csv
│   └── sample_submission.csv
```

## ⚙️ Pipeline Execution Guide

After the datasets are placed according to instructions above the following steps should be done to execute the full pipeline.

### 🧹 Data Preprocessing

To clean and prepare the dataset for training, run the following command from the root directory of the project:

```bash
python src/data_preprocessing.py
```

This script will load the raw dataset from the `data/` folder, perform necessary preprocessing steps, and save the processed data as a `.pkl` file in the same directory for use in model training.

### 🎯 Model Training

The model can be trained in two different ways depending on your setup and preference:

Locally – using your Python environment.

With Docker – using a fully containerized environment.

#### 🖥️ Local Training

To train the model using your local environment:

```bash
python src/train/train.py
```

#### 🐳 Training with Docker

Firstly, Build the Docker image:

```bash
docker build -f src/train/Dockerfile -t my-train-image .
```

Run the training container:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/logs:/app/logs" \
  -v "$(pwd)/outputs:/app/outputs" \
  my-train-image
```

### 🔍 Inference & Evaluation

Model Inference can also be done in two different ways.

#### 🖥️ Local Inference

```bash
python src/inference/inference.py
```

#### 🐳 Inference with Docker

Build the Docker image first:

```bash
docker build -f src/inference/Dockerfile -t my-inference-image .
```

Run the inference container:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/outputs:/app/outputs" \
  my-inference-image
```

## ✅ Wrap Up

![image](https://github.com/user-attachments/assets/05f55f84-3602-4cb1-a708-d29f1ef2343c)

This image shows the final evaluation results of the submission generated by this repository's LightGBM model pipeline, using the exact data preprocessing, feature engineering, and training approach described above. On the photo there are custom evaluation metrics (introduced in the [competitors guide](https://github.com/Mcompetitions/M5-methods/blob/master/M5-Competitors-Guide.pdf)), used by competition organizers to rank participants' models.

The core objective of this project was to build a fast yet strong forecasting model that could support both real-time experimentation and business decision-making. Model speed was crucial for:

- Iterative experimentation and validation

- Faster inference for large-scale retail datasets

- Convenient integration into production pipelines

**How We Achieved This?**

- Moderate preprocessing: Balanced approach to data cleaning—enough to improve model accuracy, but light enough to avoid unnecessary computation.

- Smart feature engineering: Thoughtful addition of features that help the model learn useful patterns without overfitting or bloating input dimensions.

- LightGBM: Chosen for its proven speed and performance among other tree-based models, especially as seen in other top Kaggle solutions.

**Result:**

This final submission achieved the following scores:

- WRMSSE: 5.39065

- Scaled Validation Score: 0.74172

At the time of submission, this placed the model in the top 12% of over 5,500 participants in the M5 Forecasting Accuracy competition, validating both the speed and reliability goals of this repository.







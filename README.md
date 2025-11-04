
---

# End-to-End MLOps: Energy Consumption Prediction

This repository contains the complete source code and configuration for an end-to-end MLOps project focused on predicting household energy consumption. The project demonstrates a full lifecycle from data preprocessing and experiment tracking to containerization and automated cloud deployment.

[![CI/CD - Build and Push Docker Image]<img width="1919" height="886" alt="Screenshot 2025-10-28 160817" src="https://github.com/user-attachments/assets/c34f1e56-9a15-448b-953f-769ab408acdb" />


## Table of Contents
- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [MLOps Pipeline Architecture](#mlops-pipeline-architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [How to Reproduce the Pipeline](#how-to-reproduce-the-pipeline)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Running Experiments](#2-running-experiments)
  - [3. Training the Final Model](#3-training-the-final-model)
- [Containerization with Docker](#containerization-with-docker)
- [CI/CD with GitHub Actions](#cicd-with-github-actions)
- [Cloud Deployment on AWS](#cloud-deployment-on-aws)

## Project Overview

The primary goal of this project is to predict the `Global_active_power` from the "Individual household electric power consumption" dataset. Beyond just building a model, this project focuses on implementing a robust, automated, and reproducible MLOps workflow.

**Key Features:**
- **Data Versioning:** Using DVC to track changes in data and models.
- **Experiment Tracking:** Leveraging DVC Experiments to systematically run and compare different models (RandomForest, XGBoost) and hyperparameters.
- **Reproducible Pipeline:** A DVC pipeline (`dvc.yaml`) ensures that every step from data processing to model evaluation is versioned and reproducible.
- **Web Application:** A Flask-based web interface to serve the final model for real-time predictions.
- **Containerization:** A Dockerfile to package the application, its dependencies, and the trained model into a portable container.
- **Automated CI/CD:** A GitHub Actions workflow that automatically builds a new Docker image and deploys it to an AWS EC2 instance on every push to the `master` branch.
- **Large File Handling:** Git LFS is configured to handle large model files that exceed GitHub's standard limits.

## Tech Stack

- **Data & Modeling:** Pandas, Scikit-learn, XGBoost
- **MLOps & Versioning:** Git, DVC, Git LFS
- **Application:** Flask, Gunicorn
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Cloud Provider:** Amazon Web Services (AWS)
  - **Hosting:** EC2
  - **Deployment:** Systems Manager (SSM)
  - **Permissions:** IAM

## MLOps Pipeline Architecture

The project follows a modern MLOps architecture:

1.  **Code & Data:** All code is versioned with Git. DVC is used to version large data files and models, which are stored in a remote storage (e.g., S3).
2.  **Experimentation:** Developers run experiments locally using `dvc exp run`. Results are tracked and compared with `dvc exp show`.
3.  **CI (Continuous Integration):** A `git push` to the `master` branch triggers the GitHub Actions workflow.
4.  **Build:** The workflow builds the Flask application into a Docker image.
5.  **CD (Continuous Delivery):** The newly built image is pushed to a Docker Hub registry.
6.  **CD (Continuous Deployment):** The workflow securely sends a command via AWS SSM to the EC2 instance.
7.  **Deployment:** The EC2 instance pulls the new Docker image from the registry and restarts the container, making the update live.

  <!-- You can create and host a simple diagram to make this more visual -->

## Getting Started

### Prerequisites
- Python 3.9+
- Git and Git LFS
- DVC (`pip install dvc`)
- Docker Desktop

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/khushinvyas/energy_consumption_prediction_mlops.git
    cd energy_consumption_prediction_mlops
    ```

2.  **Install Git LFS:**
    ```bash
    git lfs install
    ```

3.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

4.  **Pull DVC-tracked data and models:**
    (Note: You would first need to configure a DVC remote storage)
    ```bash
    dvc pull
    ```

## Project Structure
```
.
├── .dvc/                   # DVC internal files
├── .github/workflows/      # GitHub Actions CI/CD pipeline
├── data/
│   ├── raw/                # Raw dataset
│   └── processed/          # Processed and split data (tracked by DVC)
├── models/                 # Trained model files (tracked by DVC & Git LFS)
├── templates/              # HTML templates for Flask app
├── app.py                  # Flask application script
├── dvc.yaml                # DVC pipeline definition
├── evaluate.py             # Model evaluation script
├── Dockerfile              # Docker configuration
├── params.yaml             # Parameters for experiments
├── preprocess.py           # Data preprocessing script
├── requirements.txt        # Python dependencies
└── train.py                # Model training script
```

## How to Reproduce the Pipeline

### 1. Data Preprocessing
This stage cleans the raw data and splits it into training and testing sets.
```bash
dvc repro preprocess
```

### 2. Running Experiments
To run all defined experiments from `params.yaml`:
```bash
dvc exp run --all
```
To view the results:
```bash
dvc exp show
```

### 3. Training the Final Model
After identifying the best experiment, apply its parameters and reproduce the full pipeline to generate the final model.
```bash
# Apply the parameters from the best experiment
dvc exp apply <experiment_name>

# Run the full pipeline
dvc repro
```

## Containerization with Docker

To build and run the application as a Docker container locally:

1.  **Build the image:**
    ```bash
    docker build -t energy-prediction-app .
    ```

2.  **Run the container:**
    ```bash
    docker run -p 5000:5000 energy-prediction-app
    ```
    The application will be accessible at `http://localhost:5000`.

## CI/CD with GitHub Actions

The workflow is defined in `.github/workflows/ci-cd.yml` and performs the following on every push to `master`:
1.  **Builds** the Docker image.
2.  **Pushes** the image to Docker Hub.
3.  **Deploys** the new image to the configured AWS EC2 instance.

**Required GitHub Secrets for Deployment:**
- `DOCKERHUB_USERNAME`: Your Docker Hub username.
- `DOCKERHUB_TOKEN`: A Docker Hub Personal Access Token.
- `AWS_ACCESS_KEY_ID`: AWS IAM User Access Key.
- `AWS_SECRET_ACCESS_KEY`: AWS IAM User Secret Key.
- `AWS_REGION`: The AWS region of the EC2 instance (e.g., `us-east-1`).
- `AWS_EC2_INSTANCE_ID`: The ID of the target EC2 instance.
- `MODEL_S3_BUCKET`: S3 bucket name containing trained models.
- `XGB_MODEL_KEY`: S3 key for `xgb_model.pkl`.
- `RF_MODEL_KEY`: S3 key for `rf_model.pkl`.
- `LGBM_MODEL_KEY`: S3 key for `lgbm_model.pkl`.

On deployment, the container downloads any missing model files from S3 at startup using these variables.

## Cloud Deployment on AWS

The application is deployed to an AWS EC2 instance. The deployment is managed by GitHub Actions using AWS Systems Manager (SSM) for a secure, keyless process.

**AWS Setup Prerequisites:**
1.  An **EC2 instance** with Docker installed.
2.  An **IAM Role** (`EC2-SSM-Role`) attached to the instance with the `AmazonSSMManagedInstanceCore` policy.
3.  An **IAM User** with `AmazonSSMFullAccess` permissions, whose credentials are used in GitHub Secrets.
4.  A **Security Group** allowing inbound traffic on port `5000` (for the app) from anywhere (`0.0.0.0/0`). Port 22 (SSH) is not required and can be closed for enhanced security.
# Link -> http://51.21.196.222/
5.  <img width="1920" height="1080" alt="Screenshot (56)" src="https://github.com/user-attachments/assets/2e1b4eab-795f-46e0-ab66-2325b9d00270" />
CI trigger: bump


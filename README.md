# ICD Coding using ChatGPT

This repository contains the code to predict ICD-10 codes from clinical notes using ChatGPT

<!-- omit in toc -->
## Table of Contents
- [ICD Coding using ChatGPT](#icd-coding-using-chatgpt)
  - [🛠️ Setup](#️-setup)
    - [Cloning the codebase](#cloning-the-codebase)
    - [Python packages](#python-packages)
    - [Environment variables](#environment-variables)
  - [💾 Dataset](#-dataset)
  - [🤖 Inference](#-inference)

## 🛠️ Setup
### Cloning the codebase

```
git clone https://github.com/aryopg/chatgpt_icd_coding.git
```

### Python packages
This codebase requires multiple dependencies.
<details>
<summary>Dependencies</summary>

```
- pip
- numpy
- pydantic
- python-dotenv
- black
- isort
- tqdm
- wandb
- pandas
- matplotlib
- scikit-learn
```
</details>

We opted in to using conda as our package manager. The following will install the necessary dependencies:
```
conda env create -f environment.yml
conda activate chatgpt_icd_coding
```

### Environment variables

There are multiple environment variables required to run the training:

- **WANDB_API_KEY**: The authorisation key to access your WandB projects
- **WANDB_PROJECT_NAME**: The name that you like for this project
- **WANDB_ENTITY**: The WandB entity that will host the project
- **HF_DOWNLOAD_TOKEN**: Download token for Huggingface
- **HF_UPLOAD_TOKEN**: Upload token for Huggingface
- **HF_USERNAME**: Your HuggingFace username

- **WANDB_API_KEY**: The authorisation key to access your WandB projects.
- **WANDB_PROJECT_NAME**: The name that you like for this project.
- **WANDB_ENTITY**: The WandB entity that will host the project.
- **AZURE_OPENAI_KEY**: The key to access the OpenAI deployment on Azure.
- **AZURE_OPENAI_ENDPOINT**: The endpoint URL to access the exact OpenAI deployment on Azure.

We use the `python-dotenv` package to load these environment variables. To set them:

```
mkdir env
nano env/.env
```

Write down all of the mentioned environment variables with the appropriate values inside that file.
Certainly, you don't have to use `nano`, as long as the file name (`env/.env`) remain the same.

## 💾 Dataset

**To be completed**

## 🤖 Inference

```
python scripts/evaluate.py --config_filepath configs/deterministic_chatgpt_mimic_iv_coding_system_user.yaml
```

**To be completed**
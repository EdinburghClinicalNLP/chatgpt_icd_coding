<!-- omit in toc -->
# ICD Coding using ChatGPT

This repository contains the code to predict ICD-10 codes from clinical notes using ChatGPT

<!-- omit in toc -->
## Table of Contents
- [🛠️ Setup](#️-setup)
  - [Cloning the codebase](#cloning-the-codebase)
  - [Python packages](#python-packages)
  - [Environment variables](#environment-variables)
- [💾 Dataset](#-dataset)
- [🤖 Inference](#-inference)
- [🔬 Evaluation](#-evaluation)

## 🛠️ Setup
### Cloning the codebase

```bash
git clone --recurse-submodules https://github.com/aryopg/chatgpt_icd_coding.git
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
- pandas
- matplotlib
- scikit-learn
```
</details>

We opted to use conda as our package manager. The following will install the necessary dependencies:
```bash
conda env create -f environment.yaml
conda activate chatgpt_icd_coding
```

### Environment variables

There are multiple environment variables required to run the training:

- **AZURE_OPENAI_KEY**: The key to access the OpenAI deployment on Azure.
- **AZURE_OPENAI_ENDPOINT**: The endpoint URL to access the exact OpenAI deployment on Azure.

We use the `python-dotenv` package to load these environment variables. To set them:

```bash
mkdir env
nano env/.env
```

Write down all of the mentioned environment variables with the appropriate values inside that file.
Certainly, you don't have to use `nano`, as long as the file name (`env/.env`) remains the same.

## 💾 Dataset

The datasets are generated using: https://github.com/joakimedin/medical-coding-reproducibility
We only used the test split for the inference.

## 🤖 Inference

To run the prediction, we need a config file that contains the hyperparameters of ChatGPT.
See the example below for a "deterministic" prediction run (`temperature == 0`, `top_p == 0`):

```bash
python scripts/inference.py --config_filepath configs/deterministic_chatgpt_mimic_iv_coding_system_user.yaml
```

## 🔬 Evaluation

After the inference, there will be an output folder containing all the predictions made by ChatGPT (e.g. `outputs/2023_08_25__09_19_37/predictions`).
We need the path to that folder as well as the path to the ground-truth test split (e.g. `data/disch_raw_test_split.csv`) to provide the true labels.

```bash
python scripts/evaluate.py --predictions_dir <PREDICTIONS_DIR> --groundtruth_path <GROUNDTRUTH_PATH>
```

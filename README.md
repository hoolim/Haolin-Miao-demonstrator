# Fine-Tuning Whisper for Cantonese-English Code-Switching

This repository documents a project focused on fine-tuning OpenAI's Whisper models (`small` and `large-v3`) to improve their performance on a Cantonese-English code-switching Automatic Speech Recognition (ASR) task.

The project covers the complete workflow, from data preprocessing and environment setup to model training, evaluation, and analysis of the results.

## Research Question

> *To what extent does fine-tuning Whisper-small and Whisper-large-v3 on Cantonese-English code-mixed speech reduce WER and CER, compared to their zero-shot performance on the same data?*

## Key Results

The experiments demonstrated that fine-tuning significantly improves model performance, particularly for the `whisper-small` model. In its zero-shot state, `whisper-small` produced a high Word Error Rate (WER) of 87.37%. After fine-tuning, its WER was more than halved, dropping to **40.79%**. The improvement in Character Error Rate (CER) was even more dramatic, falling from 47.23% to just **5.90%**.

The larger `whisper-large-v3` model also benefited from fine-tuning, though it showed signs of overfitting. Its WER improved from a zero-shot baseline of 82.62% to **67.47%**, and its CER was reduced from 35.12% to **23.21%**. These results suggest that for this dataset, the smaller model provided a better balance of adaptation and generalization.

## Dataset & License

The MCE (Mandarin-Cantonese-English) dataset used for this project is publicly available for research purposes.

* **Download Link:** [**MCE Dataset via Google Drive**](https://drive.google.com/file/d/1CFgHxTzYBKnIkRVBdCwlJXahZq3Zi87B/view)

#### 📄 License

The MCE dataset is intended solely for research purposes to support advancements in automatic speech recognition (ASR). Any commercial use without prior authorization is strictly prohibited. For potential collaborations or licensing inquiries, please reach out to 📮 `shelton1013@outlook.com`.



## Getting Started

#### 1. Environment Setup

This project uses **Miniconda** for environment management to ensure reproducibility.

```bash
# Step 1: Create a new Conda environment named whisper_env
conda create --name whisper_env python=3.9 -y

# Step 2: Activate the new environment
conda activate whisper_env

# Step 3: Install all required Python packages
# First, install PyTorch (for CUDA 12.1)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install Transformers and other tools
pip install transformers[torch] datasets jiwer accelerate pandas tqdm tensorboard
pip install librosa soundfile gradio

# Finally, install ffmpeg
conda install -c conda-forge ffmpeg
```

#### 2. Data Preprocessing

Before training, run the preprocessing script to generate a unified metadata file.

* **Important**: Open the `prepare_dataset.py` script and modify the `DATASET_BASE_PATH` variable to point to the absolute path of your MCE dataset.

```bash
python prepare_dataset.py
```
This command will create a `metadata.csv` file in the project's root directory.

#### 3. Model Fine-Tuning

The `finetune_whisper.py` script is used for model training. You can configure the `MODEL_NAME` and other hyperparameters (e.g., `learning_rate`) directly within the script to select the model and strategy.

* **Recommended launch command (runs in the background)**:
  ```bash
  # Ensure finetune_whisper.py is configured as needed
  nohup python -u finetune_whisper.py > training.log 2>&1 &
  ```
* **Monitor training progress**:
  ```bash
  tail -f training.log
  ```


After training is complete, you can evaluate performance and prepare the model for sharing.


## Repository Structure

* `prepare_dataset.py`: Parses the raw MCE dataset and generates `metadata.csv`.
* `finetune_whisper.py`: The core script for fine-tuning a chosen Whisper model.
* `README.md`: This documentation file.

## Acknowledgements

This work was inspired by the official Whisper fine-tuning guides and resources provided by the HuggingFace team.

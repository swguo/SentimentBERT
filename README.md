# Sentiment Analysis with BERT

This project implements a sentiment analysis system using a pre-trained BERT model. It classifies text into three sentiment categories: **negative**, **neutral**, and **positive**. The system is designed for training, inference, and evaluation, and it includes an interactive web interface powered by `gradio`.

## Features

- Train a sentiment analysis model on labeled data.
- Save and load trained models for inference and evaluation.
- Perform inference on test datasets with results saved to a CSV file.
- Evaluate predictions with a confusion matrix and classification report.
- Interactive web interface for real-time sentiment classification.

## Directory Structure

```
.
├── demo.ipynb           # Gradio interface for real-time prediction
├── data_processing.py   # Data loading and preprocessing
├── dataset.py           # Dataset preparation and DataLoader implementation
├── eval.py              # Evaluate inference results with confusion matrix
├── inference.py         # Perform inference on test data
├── main.py              # Main script for training the model
├── model.py             # BERT-based classifier model
├── train.py             # Training and validation functions
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── saved_models/
    └── best_model_state.bin  # Trained model weights
└── data/
    └── train.csv         # Training data
    └── test.csv          # Test data
    └── infr_result.csv   # Inference results
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/swguo/SentimentBERT.git
   cd SentimentBERT
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the Model

To train the sentiment analysis model:

```bash
python main.py --data_path 'data/reviews.csv' \
               --model_name 'bert-base-uncased' \
               --max_len 160 \
               --batch_size 16 \
               --epochs 4 \
               --learning_rate 2e-5
```

The best model will be saved to `best_model_state.bin`.

### 2. Perform Inference

To perform inference on a test dataset:

```bash
python inference.py --test_data_path 'data/test.csv' \
                    --model_path 'model/best_model_state.bin' \
                    --output_path 'data/infr_result.csv'
```

The results will be saved to `data/infr_result.csv`.

### 3. Evaluate the Model

To evaluate the inference results:

```bash
python eval.py --inference_results_path 'data/infr_result.csv' \
               --class_names negative neutral positive 
```

This will display the confusion matrix and classification report.

### 4. Interactive Web Interface

To use the real-time sentiment analysis interface in gradio:

```bash
demeo.ipynb
```

Access the interface in your browser at the URL provided by `gradio`.

## Example Data Format

The input CSV file (`data/reviews.csv`) should have the following structure:

| content                           | score |
|-----------------------------------|-------|
| "The product is amazing!"         | 5     |
| "This product is so so."          | 3     |
| "Absolutely terrible experience." | 1     |

- `content`: The review text.
- `score`: The review rating (used to derive sentiment labels).

## Example Trainging Data Format

The input CSV file (`data/train.csv`) should have the following structure:

| content                           | sentiment |
|-----------------------------------|-------|
| "The product is amazing!"         | 0 : negative     |
| "This product is so so."          | 1 : neutral      |
| "Absolutely terrible experience." | 2 : positive     |

## Dependencies

Key dependencies are listed in `requirements.txt`. Notable libraries include:

- `transformers` (Hugging Face)
- `torch` (PyTorch)
- `gradio`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Install them with:

```bash
pip install -r requirements.txt
```

## Results

The model achieves the following results on the validation set:

- **Accuracy**: 87%
- **F1 Score**: 85%

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.# SentimentBERT

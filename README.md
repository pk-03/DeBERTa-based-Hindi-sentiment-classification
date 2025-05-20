
# Hindi Sentiment Classification using DeBERTa

This project fine-tunes the `microsoft/deberta-v3-base` transformer model for sentiment classification on Hindi text. The goal is to classify Hindi reviews into one of three sentiment classes: **positive**, **negative**, or **neutral**.

---

## 🧠 Model Architecture

- **Model**: [DeBERTa v3 Base](https://huggingface.co/microsoft/deberta-v3-base)
- **Tokenizer**: AutoTokenizer from Hugging Face
- **Framework**: PyTorch with HuggingFace Transformers & Trainer API

---

## 📁 Dataset Format

The dataset should be a `.csv` or `.tsv` file with the following two columns:

- `Reviews`: Hindi review text.
- `labels`: Sentiment labels (e.g., `"positive"`, `"negative"`, `"neutral"`)

Example:

| Reviews                  | labels   |
|--------------------------|----------|
| यह फोन बहुत अच्छा है।   | positive |
| बैटरी खराब है।          | negative |
| यह एक औसत उत्पाद है।     | neutral  |

---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/hindi-sentiment-deberta.git
cd hindi-sentiment-deberta
```

### 2. Install Dependencies
Make sure Python 3.8+ is installed.

```bash
pip install transformers torch pandas scikit-learn indic-nlp-library
```

---

## 🏋️‍♀️ Training the Model

### 1. Load and preprocess your dataset
Ensure your `DataFrame` has `Reviews` and `labels` columns.

```python
data = pd.read_csv("your_dataset.csv")  # or .tsv
```

### 2. Run the training script
The script:
- Encodes text with DeBERTa tokenizer
- Prepares datasets
- Trains using HuggingFace `Trainer`
- Evaluates on a held-out test set

```python
python train_sentiment_model.py
```

> **Note:** You can customize training parameters such as `batch_size`, `epochs`, and `learning_rate` inside the script.

---

## 🧪 Evaluation

After training, the model is evaluated using the following metrics:

- Accuracy
- F1 Score (weighted)
- Precision (weighted)
- Recall (weighted)

The best model (based on F1 score) is automatically saved and restored for evaluation.

---

## 🧠 Inference

Use the `classify_text` function to predict sentiment for new Hindi inputs:

```python
sample = "यह उत्पाद बहुत बेकार है।"
predicted_label = classify_text(sample)
print(f"Predicted Sentiment: {predicted_label}")
```

---

## 💾 Saving and Loading the Model

To save the fine-tuned model and tokenizer:

```python
model.save_pretrained("./deberta-hindi-sentiment")
tokenizer.save_pretrained("./deberta-hindi-sentiment")
```

To load it again later:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./deberta-hindi-sentiment")
tokenizer = AutoTokenizer.from_pretrained("./deberta-hindi-sentiment")
```

---

## 📊 Results

| Metric    | Value |
|-----------|-------|
| Accuracy  |  -    |
| F1 Score  |  -    |
| Precision |  -    |
| Recall    |  -    |

> You can update this table after training.
<!-- Test Results: {'eval_loss': 1.5190038681030273, 'eval_accuracy': 0.8454301075268817, 'eval_f1': 0.8451949511547825, 'eval_precision': 0.847356804309445, 'eval_recall': 0.8454301075268817, 'eval_runtime': 15.0775, 'eval_samples_per_second': 98.69, 'eval_steps_per_second': 12.336, 'epoch': 38.0} -->

---

## 📌 Future Work

- Add support for class imbalance handling.
- Incorporate validation split for hyperparameter tuning.
- Integrate IndicNLP for preprocessing (normalization, sentence splitting).

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

## 🙏 Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Microsoft DeBERTa](https://github.com/microsoft/DeBERTa)
- [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library)

---

## ✨ Contact

For queries, reach out to [yourname@example.com] or create an issue.

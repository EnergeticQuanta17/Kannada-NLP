import torch
from datasets import load_dataset, load_metric
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Koodsml/KooBERT")
model = AutoModelForSequenceClassification.from_pretrained("Koodsml/KooBERT", num_labels=2)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=128)
# Load the CoLA dataset
dataset = load_dataset("glue","cola")
dataset = dataset.rename_column('sentence', 'text')
datset_tok = dataset.map(tokenize_function, batched=True)
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Define the training arguments
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datset_tok['train'],
    eval_dataset=datset_tok['validation'],
    compute_metrics=compute_metrics,
)
# Fine-tune on the CoLA dataset
trainer.train()
# Evaluate on the CoLA dataset
eval_results = trainer.evaluate(eval_dataset=cola_dataset['validation'])
print(eval_results)

from sentence_transformers import SentenceTransformer
# Load the KooBERT model
koo_model = SentenceTransformer('Koodsml/KooBERT', device="cuda")
# Define the text
text = "ರಾಮ ಮನೆಗೆ ಹೋದ"
# Get the embedding
embedding = koo_model.encode(text)
print(embedding)
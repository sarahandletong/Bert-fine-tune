
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np

#  Add the previous saved model and tokenizer
model_path = "./bert-chnsenticorp-finetuned"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the dataset and perform word segmentation
dataset = load_dataset("lansinuote/ChnSentiCorp")

def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

encoded_dataset = dataset.map(preprocess_function, batched=True)
eval_dataset = encoded_dataset["test"]

# Defining evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

# Define training parameters (only for evaluation, not training)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=32,
)

# Create Trainer and evaluate
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Evaluate and print accuracy again
eval_result = trainer.evaluate()
print(f"Validation set accuracy: {eval_result['eval_accuracy']:.4f}")

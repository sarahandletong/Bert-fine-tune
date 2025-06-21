

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

# Loading the dataset from hugingface
dataset = load_dataset("lansinuote/ChnSentiCorp")

# Chinese BERT word segmenter and model
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# data preprocessing: word segmentation
def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# setting up training and validation sets
train_dataset = encoded_dataset["train"]
eval_dataset = encoded_dataset["test"]

# defining evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

# Configure training parameters
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Evaluate
eval_result = trainer.evaluate()
print(f"Validation set accuracy: {eval_result['eval_accuracy']:.4f}")



# Save the model and tokenizer uniformly
model_path = "./bert-chnsenticorp-finetuned"
trainer.save_model(model_path) # Save model weights, etc.
tokenizer.save_pretrained(model_path)   # Store word segmentation configuration, etc.

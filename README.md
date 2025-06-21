# Bert-fine-tune
## Bert Chinese Sentiment Classification
This project uses the Hugging Face Transformers framework to fine-tune the Chinese sentiment classification dataset, and the model is based on bert-base-chinese.

#### Script description: bert_chinese_sentiment.py
#### The main process of this script includes:

#### 1. Loading dataset
Load the Chinese sentiment classification dataset lansinuote/ChnSentiCorp from Hugging Face Hub.

#### 2. Loading pre-trained model and word segmenter
Use Hugging Face's bert-base-chinese pre-trained model and corresponding word segmenter.

#### 3. Data preprocessing
Word segmentation, truncation and padding of text, and the maximum sequence length is set to 128.

#### 4. Training configuration
Set training parameters such as learning rate, batch size, number of training rounds, weight decay, etc., and specify to evaluate and save the model at the end of each round.

#### 5. Model training and evaluation
Use Trainer API to fine-tune the model and calculate the accuracy of the validation set.

#### 6. Save the model
After training, save the fine-tuned model and word segmenter to the local directory ./bert-chnsenticorp-finetuned.

#### 7. Running environment

Transformers library

Datasets library

Numpy

#### 8. Results
The model achieved a high accuracy rate (about 93.8%) on the validation set.

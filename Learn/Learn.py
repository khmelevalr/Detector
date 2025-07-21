from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import numpy as np
import os
 
class CustomDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.texts = []
        self.labels = []
        for label in ["0", "1"]:
            folder_path = os.path.join(data_dir, label)
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                        text = file.read()
                        self.texts.append(text)
                        self.labels.append(int(label))
 
        self.tokenizer = tokenizer
        self.max_length = max_length
 
    def __len__(self):
        return len(self.texts)
 
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
 
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item
 
data = "C:/dataset"
 
model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name, model_max_length=512)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
 
custom_dataset = CustomDataset(data, tokenizer)
 
train_data, val_data = train_test_split(custom_dataset, test_size=0.1, random_state=42)
train_dataset = torch.utils.data.Subset(custom_dataset, indices=list(range(len(train_data))))
val_dataset = torch.utils.data.Subset(custom_dataset, indices=list(range(len(val_data))))
 
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir='./logs',
    logging_steps=100,
    seed=42
)
 
import evaluate
accuracy_metric = evaluate.load('accuracy')
 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)
 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
 
trainer.train()
trainer.save_model("./detector1")
tokenizer.save_pretrained("./detector1")
 
print("Модель успешно сохранена в папку 'detector1'")

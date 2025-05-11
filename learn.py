import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import torch
import pymorphy3
from pymorphy3 import MorphAnalyzer
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
file_path = "C:/Users/max/Desktop/Sturdy/3_curs_2_sem/КурсоваяРабота2Семестр/dataset1.xlsx"
df = pd.read_excel(file_path)
df.columns = ['text', 'label']
texts = df['text'].tolist()
labels =  df['label'].tolist()

# оставляем только обучающую часть, это 80% от всего корпуса данных
texts = [text for i, text in enumerate(texts, start=1) if i % 9 != 0 and i % 10 != 0]
labels = [label for i, label in enumerate(labels, start=1) if i % 9 != 0 and i % 10 != 0]



morph = MorphAnalyzer()
texts_lemmatized = []
for i in range(0, len(texts)):
    tokens = word_tokenize(texts[i])
    lems = []
    for token in tokens:
        tag = morph.parse(token)[0].tag
        if 'PNCT' in tag:
            continue
        lems.append(morph.parse(token)[0].normal_form)
    texts_lemmatized.append(" ".join(lems))


# model_name = "DeepPavlov/rubert-base-cased"
model_name = "sberbank-ai/ruRoberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)



print("Инициализация токенизатора успешно")
encodings = tokenizer(texts_lemmatized, truncation=True, padding=True, max_length=192)

dataset = Dataset.from_dict({
        "input_ids": encodings['input_ids'],
        "attention_mask": encodings['attention_mask'],
        "labels": labels
})

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
training_args = TrainingArguments(
    output_dir="C:/Users/max/Desktop/Sturdy/3_curs_2_sem/КурсоваяРабота2Семестр/output_dir/results_sberv2",
    evaluation_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=4,#2,3,4,5,6
    weight_decay=0.01,
    logging_dir="C:/Users/max/Desktop/Sturdy/3_curs_2_sem/КурсоваяРабота2Семестр/logging_dir/logs_sberv2",
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained("C:/Users/max/Desktop/Sturdy/3_curs_2_sem/КурсоваяРабота2Семестр/trained_model_rurobertalarge_4epochs")
tokenizer.save_pretrained("C:/Users/max/Desktop/Sturdy/3_curs_2_sem/КурсоваяРабота2Семестр/trained_model_rurobertalarge_4epochs")
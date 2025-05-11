from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import pymorphy3
from pymorphy3 import MorphAnalyzer
import nltk
from nltk.tokenize import word_tokenize
import time

#trained_model_rurobertalarge_4epochs
#заменяем на модель, которую хотим протестировать
model_path = "C:/Users/max/Desktop/Sturdy/3_curs_2_sem/КурсоваяРабота2Семестр/trained_model_rubert_3epochs"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

file_path = "C:/Users/max/Desktop/Sturdy/3_curs_2_sem/КурсоваяРабота2Семестр/dataset1.xlsx"
df = pd.read_excel(file_path)
df.columns = ['text', 'label']
texts = df['text'].tolist()

labels =  df['label'].tolist()
texts = [text for i, text in enumerate(texts, start=1) if i % 9 == 0 or i % 10 == 0]
labels = [label for i, label in enumerate(labels, start=1) if i % 9 == 0 or i % 10 == 0]

morph = MorphAnalyzer()
texts_lemmatized = []
for i in range(0, len(texts)):
    tokens = word_tokenize(texts[i])
    lems = []
    for token in tokens:
        tag = morph.parse(token)[0].tag
        if 'PNCT' in tag:
            continue
        lems.append(token)
    texts_lemmatized.append(" ".join(lems))

#positive - ргументация есть
TP = 0
TN = 0
FP = 0
FN = 0
start_time = time.time()
for i in range(0, len(texts_lemmatized)):
    test_text = texts_lemmatized[i]
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    print("Prediction (0 or 1):", predictions.item(), " label = ", labels[i], "i = ", i+1)
    if labels[i] == 1:
        if labels[i] == predictions.item():
            TP += 1
        else:
            FN += 1
    else:
        if labels[i] == predictions.item():
            TN += 1
        else:
            FP += 1
end_time = time.time()
#TP -правильные 1
#TN - правильные 0
print("TP = ", TP, "TN = ", TN, "FP = ",FP, "FN = ", FN)
acc = (TP+TN)/(TP+TN+FP+FN)
print("Accuracy = ", acc * 100, "%")
print("Error Rate = ", (1 - acc) * 100, "%")
p = TP/(TP+FP)
print("Precision = ", p )
r = TP/(TP+FN)
print("Recall = ", r )
print("F-Measure = ", (2*p*r)/(p+r))
print("Time in seconds = ", end_time - start_time)

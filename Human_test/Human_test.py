import re
from transformers import pipeline
from transformers import BertTokenizer, BertForNextSentencePrediction 
import torch


def function1(sentences):
    amount = 0
    divergence = 0
    count = len(sentences)
    for sentence in sentences:
        tokens = sentence.split()
        if '-' in tokens:
            amount += len(tokens)-1
        else:
            amount += len(tokens)
    mean = amount/count  #среднее арифметическое
    for sentence in sentences:
        tokens = sentence.split()
        if tokens:
            if '-' in tokens:
                divergence += (len(tokens)-1 - mean)**2
            else:
                divergence += (len(tokens) - mean)**2
    result_divergence = (divergence/count)**(1/2) #средняя квадратичная ошибка
    if result_divergence <= 7.93:
        return 1
    else:
        return 0

def length(sentences):
    sum_len = 0
    count = 0
    for sentence in sentences:
        tokens = sentence.split()
        for token in tokens:
            if len(token) > 2:
                sum_len += len(token)
                count += 1
    result_len = sum_len/count
    if result_len <= 7.67:
        return 1
    else:
        return 0


def emotions(text):
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    if label == 'neutral':
        result_emotions = score
    else:
        result_emotions = 1 - score
    if result_emotions >= 0.20:
        return 1
    else:
        return 0
    
def connection(sentences):
    model = 'DeepPavlov/rubert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(model)
    model = BertForNextSentencePrediction.from_pretrained(model)
    connection_list=[]
    for i in range(len(sentences)-1):
        tokens = tokenizer(sentences[i],sentences[i+1],return_tensors='pt')
        outputs = model(**tokens)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        is_next_prob = probs[0][0].item()
        connection_list.append(is_next_prob)
    s=0
    for j in connection_list:
        s+=j
    connection_result = s/len(connection_list)
    if connection_result<0.95:
        return 1
    else:
        return 0

def number_of_words(sentences):
    k = 0
    sum = 0
    for sentence in sentences:
        tokens = sentence.split()
        k += 1
        sum += len(tokens)
    sum_result = sum/k
    if sum_result < 17.37:
        return 0, sum
    else:
        return 1, sum

def tabs(text):
    tab=0
    sum_number_of_words=0
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sum_number_of_words = number_of_words(sentences)[1]
    block = text.split('\n')
    for j in block:
        tab+=1
    res = tab/sum_number_of_words
    if res > 0.03:
        return 1
    else:
        return 0

def tires(text):
    tire = text.count("\u2014")+text.count("\u2013")
    sum_number_of_words = number_of_words(sentences)[1]
    tire = tire/sum_number_of_words
    if tire>0.0071:
        return 1
    else:
        return 0

def quotes(text):
    quote = text.count("\"")/2
    sum_number_of_words = number_of_words(sentences)[1]
    quote = quote/sum_number_of_words
    if quote>0.0057:
        return 1
    else:
        return 0

def brackets(text):
    bracket = text.count("(")
    sum_number_of_words = number_of_words(sentences)[1]
    bracket = bracket/sum_number_of_words
    if bracket<0.0010:
        return 0
    else:
        return 1

def slashs(text):
    slash = text.count("/")
    if slash>=1:
        return 0
    else:
        return 1

def points(text):
    point = text.count(":")
    sum_number_of_words = number_of_words(sentences)[1]
    point = point/sum_number_of_words
    if point>=0.009:
        return 0
    else:
        return 1

def keywords(text):
    k1="Таким образом"
    k1_="таким образом"
    k2="Кроме того"
    k2_="кроме того"
    k3='Все вышеперечисленное'
    k3_='все вышеперечисленное'
    k3__='Всё вышеперечисленное'
    k3___='всё вышеперечисленное'
    k4='Благодаря'
    k4_='благодаря'
    k5 = 'заключается'
    c = text.count(k1)+text.count(k1_)+text.count(k2)+text.count(k2_)+text.count(k3)+text.count(k3_)+text.count(k3__)+text.count(k3___)+text.count(k4_)+text.count(k4_)+text.count(k5)
    if c > 3:
        return 1
    if c==2:
        return 0.9
    if c==1:
        return 0.8
    else:
        return 0


classifier = pipeline("text-classification", model="blanchefort/rubert-base-cased-sentiment")
file = open(r'C:\dataset\test_text.txt', 'r', encoding='utf-8')
text = file.read()
sentences = re.split(r'[.!?]', text)
sentences = [s.strip() for s in sentences if s.strip()]

k1 = function1(sentences)
k2 = length(sentences)
k3 = number_of_words(sentences)[0]
k4 = tabs(text)
k5 = emotions(text)
k6 = connection(sentences)
k7 = tires(text)
k8 = quotes(text)
k9 = brackets(text)
k10 = slashs(text)
k11 = points(text)
k12 = keywords(text)
w1 = 1
w2 = 1
w3 = 1
w4 = 1
w5 = 0.3
w6 = 1
w7 = 1
w8 = 1
w9 = 1
w10 = 1
w11 = 1
w12 = 1

result = (k1*w1 + k2*w2 + k3*w3 + k4*w4 + k5*w5 + k6*w6 + k7*w7 + k8*w8 + k9*w9 + k10*w10 + k11*w11 + k12*w12)/(w1+w2+w3+w4+w5+w6+w7+w8+w9+w10+w11+w12)

print(k1)
print(k2)
print(k3)
print(k4)
print(k5)
print(k6)
print(k7)
print(k8)
print(k9)
print(k10)
print(k11)
print(k12)


if result>0.5:
    print(f'Текст сгенерирован ИИ с вероятностью {(result*100):.2f} %')
else:
    print(f'Текст написан человеком c вероятностью {((1-result)*100):.2f} %')
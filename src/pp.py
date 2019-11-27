import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from konlpy.tag import Twitter
import pandas as pd

keyword='립스틱'

def delSpace(string):
    return ' '.join(string.strip().split())

%matplotlib inline
data=pd.read_excel('{}.xlsx'.format(keyword))
for i in range(len(data['본문'])):
    data['본문'][i]=data['본문'][i].replace('#', ' ')
    data['댓글'][i]=data['댓글'][i].replace('#', ' ')
text=[]
coment=[]
for i in range(len(data)):
    text.append(data['본문'][i])
    coment.append(data['댓글'][i])
    
hangul = re.compile('[^ ㄱ-ㅣ가-힣a-zA-Z0-9]+')
ttext=[]
ccoment=[]
for i in range(len(text)):
    result = hangul.sub('', text[i])
    ttext.append(result)
for j in range(len(coment)):
    result = hangul.sub('', text[i])
    ccoment.append(result)
for i in range(len(ttext)):
    ttext[i]=delSpace(ttext[i])
for j in range(len(ccoment)):
    ccoment[j]=delSpace(ttext[i])
text=' '.join(ttext)
comment=' '.join(ccoment)


sentance = text
twt = Twitter()
tagging = twt.pos(sentance)
print(tagging)##품사 뽑기

temp=[]
twt=Twitter()
temp=twt.nouns(text)

def flatten(l):
    flatList=[]
    for elem in l:
        if type(elem)==list:
            for e in elem:
                flatList.append(e)
        else:
            flatList.append(elem)
    return flatList
word_list=flatten(temp)
word_list=pd.Series([x for x in word_list if len(x)>1])
print(word_list.value_counts())##명사 빈도 추출
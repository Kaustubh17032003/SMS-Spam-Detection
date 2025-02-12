import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

def text_prep(x):
    # LowerCasing
    x=x.lower()
    # Tokenization
    x=nltk.word_tokenize(x)
    y=[]
    # Removing Special Characters
    for i in x:
        if i.isalnum():
            y.append(i)
    z=y[:]
    y.clear()
    # Removing Stopwords & Punctuations
    for i in z:
        if i not in stopwords.fileids() and i not in string.punctuation:
            y.append(i)
    # Stemming
    z.clear()
    z=y[:]
    y.clear()
    for i in z:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf=pickle.load(open("vectorizer.pkl","rb"))
model=pickle.load(open("model.pkl","rb"))

st.title("SMS Spam Classifier")
input_sms=st.text_area("Enter the SMS")

if st.button("predict"):
    # 1. preprocess
    prep_sms=text_prep(input_sms)
    # 2. Vectorize
    vector=tfidf.transform([prep_sms])
    # 3. Predict
    result=model.predict(vector)[0]
    # 4. Display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")
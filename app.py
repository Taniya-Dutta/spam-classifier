import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

nltk.download('punkt')
cv=pickle.load(open('true_vectorizers.pkl','rb'))
model=pickle.load(open('true_models.pkl','rb'))

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(text)
st.title('Email/SMS spam classifier')
input_sms=st.text_input('Enter the message')
if st.button('Predict'):
   transformed_sms=transform_text(input_sms)
   vector_input=cv.transform([transformed_sms])
   res=model.predict(vector_input)[0]
   if res==0:
      st.header('not spam')
   else:
      st.header('spam')
# to  make the website in terminal type  streamlit run app.py






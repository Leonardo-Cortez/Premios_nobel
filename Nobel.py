import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression


st.write(''' # Predicción de categoría de Premio Nobel ''')
st.image("Nobel.png", caption="Su creador fue el inventor sueco Alfred Nobel mediante su testamento en 1895.")

st.header('Texto')


def user_input_features():

    # Entrada
    texto = st.text_input("Introduce el texto a evaluar")

    user_input_data = {'Motivation_Clean': texto}

    features = pd.DataFrame(user_input_data, index=[0])

    return features


df = user_input_features()


nobel = pd.read_csv('datos_nobel_etl.csv',encoding='utf-8')

X = nobel.Motivation_Clean
y = nobel.Category

vect = CountVectorizer(stop_words='english')
X_counts = vect.fit_transform(X)

tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X_counts)

modelo = LogisticRegression(max_iter=1000,random_state=42)
modelo.fit(X_tfidf, y)


# Transformar texto introducido por el usuario
df_counts = vect.transform(df['Motivation_Clean'])
df_tfidf = tfidf.transform(df_counts)


# Predicción
prediction = modelo.predict(df_tfidf)
st.subheader('Predicción')

st.write(prediction[0])

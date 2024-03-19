import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

ps = PorterStemmer()
vector = TfidfVectorizer()

vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))


def stemming(title):
    stemmed_title = re.sub('[^a-zA-Z]', ' ', title)
    stemmed_title = stemmed_title.lower()
    stemmed_title = stemmed_title.split()
    stemmed_title = [ps.stem(word) for word in stemmed_title if not word in stopwords.words('english')]
    stemmed_title = ' '.join(stemmed_title)
    return stemmed_title


def fake_news(news):
    news = stemming(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction


if __name__ == '__main__':
    st.title('Fake News Classification app')
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here", "Some news", height=200)
    predict_btt = st.button("Predict")
    if predict_btt:
        prediction_class = fake_news(sentence)
        print(prediction_class)
        if prediction_class == [1]:
            st.success('The News is Real')
        if prediction_class == [0]:
            st.warning('The News is Fake')
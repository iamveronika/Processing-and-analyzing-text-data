import re
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from textblob import TextBlob
import nltk
import pandas as pd


data = pd.read_csv("twitter1.csv", names=['id', 'title', 'mood','text'])

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
def preproc_doc(doc):
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in
    stop_words]
    doc = ' '.join(filtered_tokens)
    return doc


data = data.dropna()
data['text'] = data['text'].apply(preproc_doc)
data['mood'] = data['mood'].apply(preproc_doc)
X = data['text']
Y = data['mood']
print(X)
print(Y)


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)


mnb_pipeline = Pipeline([('tfidf', TfidfVectorizer()),('mnb', MultinomialNB(alpha=1))])
mnb_pipeline.fit(X_train, y_train)
predict = mnb_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predict)
print('Test Accuracy :', accuracy)

cm = confusion_matrix(y_test,predict, labels=mnb_pipeline.classes_)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mnb_pipeline.classes_)
disp.plot(cmap='summer')
plt.show()


indexIrr = data[data['mood'] == 'Irrelevant'].index
d = data.drop(indexIrr)
X = d['text']
Y = d['mood']

sentiment_polarity =[TextBlob(review).sentiment.polarity for review in X]
predicted_sentiments = ['positive' if score >= 0.15 else 'neutral' if score >= 0 else 'negative' for score in sentiment_polarity]
accuracy = accuracy_score(Y, predicted_sentiments)
print('Test Accuracy :', accuracy)

labels = ['positive', 'neutral', 'negative']
cm = confusion_matrix(Y, predicted_sentiments, labels=labels)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='plasma')
plt.show()


d3 = data.sample(n=3, random_state=3)
predict = mnb_pipeline.predict(d3['text'])
d3['mnb'] = predict
sentiment_polarity =[TextBlob(review).sentiment.polarity for review in d3['text']]
predicted_sentiments = ['positive' if score >= 0.15 else 'neutral' if score >= 0 else 'negative' for score in sentiment_polarity]
d3['tb'] = predicted_sentiments
print(d3.to_string())

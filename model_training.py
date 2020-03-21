from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Prevent future/deprecation warnings from showing in output
import warnings
warnings.filterwarnings(action='ignore')
#! reading frm csv and fetching 200k rows
df = pd.read_csv('clean_final.csv')
d1 = df.head(100000)
d2 = df.tail(100000)
dff = pd.concat([d1, d2])
dff.to_csv("small_data.csv", index=False)
df = pd.read_csv("small_data.csv")

X = df.text
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = X_train.apply(lambda x: np.str_(x))
X_test = X_test.apply(lambda x: np.str_(x))
y_train = y_train.apply(lambda x: np.str_(x))
y_test = y_test.apply(lambda x: np.str_(x))

#! vectorizing
vect = CountVectorizer(max_features=1000, binary=True)
X_train_vect = vect.fit_transform(X_train)
counts = df.target.value_counts()

#! training model
sm = SMOTE()
X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)

nb = MultinomialNB()
nb.fit(X_train_res, y_train_res)
nb.score(X_train_res, y_train_res)
X_test_vect = vect.transform(X_test)
y_pred = nb.predict(X_test_vect)

print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#! saving model and vectorizer
with open("model1.pkl", 'wb') as f:
    pickle.dump(nb, f)
with open("vect.pkl", 'wb') as v:
    pickle.dump(vect, v)

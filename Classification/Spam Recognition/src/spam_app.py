import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import joblib
import os
from func import report, TSNE_plot
import gradio as gr
import seaborn as sns
from sklearn.manifold import TSNE as SKlearnTSNE
import matplotlib
matplotlib.use('Agg')


def load_data():
    data = pd.read_csv("C:\\Users\\User\\PyCharmMiscProject\\spam_Emails_data.csv", sep = ',')
    data = data.dropna(subset = ['label', 'text']).reset_index(drop = True)

    data['text'] = data['text'].str.lower()
    pd.set_option('future.no_silent_downcasting', True)
    data['label'] = data['label'].replace({'Spam':0, 'Ham':1})

    return data

def train_pipeline():
    data = load_data()
    X = data['text']
    Y = data['label']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

    Y_train = Y_train.astype(int)
    Y_test = Y_test.astype(int)

    model = make_pipeline(TfidfVectorizer(), LinearSVC(dual = "auto"))
    model.fit(X_train, Y_train)
    os.makedirs("pictures", exist_ok=True)

    report(model, X_test, Y_test)

    plt.tight_layout()
    plt.savefig("pictures/confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("📊 Confusion matrix saved to 'pictures/confusion_matrix.png'")

    joblib.dump(model, "LinearSVC.pkl")

    vectorizer = model.named_steps['tfidfvectorizer']
    joblib.dump(vectorizer, "vectorizer.pkl")
    return model, X_train, Y_train

def TSNE(model, X_train, Y_train):
    os.makedirs("pictures", exist_ok=True)
    vectorizer = model.named_steps['tfidfvectorizer']

    plt.figure(figsize=(15, 5))
    data_1000 = X_train.iloc[0:1000]
    labels_1000 = Y_train.iloc[0:1000]
    X_vec = vectorizer.transform(data_1000)

    plt.subplot(1, 2, 1)
    tsne_model = SKlearnTSNE(n_components=2, random_state=0, init='random', perplexity=30, n_iter=300)
    tsne_data = tsne_model.fit_transform(X_vec.toarray())

    tsne_df = pd.DataFrame(np.hstack((tsne_data, labels_1000.values.reshape(-1, 1))), columns=("Dim_1", "Dim_2", "label"))
    sns.scatterplot(data=tsne_df, x='Dim_1', y='Dim_2', hue='label', palette="bright")
    plt.title("Data distribution before LinearSVC")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.subplot(1, 2, 2)
    TSNE_plot(model.named_steps['linearsvc'], X_vec, labels_1000, "Data distribution after LinearSVC")

    plt.tight_layout()
    plt.savefig("pictures/TSNE_plot.png", dpi=100, bbox_inches="tight")
    print("📊 TSNE plot saved to 'pictures/TSNE_plot.png'")

def load_model():
    try:
        model = joblib.load("LinearSVC.pkl")
        print("Loaded saved model")
    except:
        print("Training model...")
        model, X_train, Y_train = train_pipeline()
        TSNE(model, X_train, Y_train)
    return model
model = load_model()

def spam_recog(text):
    prediction = model.predict([text])[0]
    return "Spam Detected" if prediction == 0 else "Spam Not Detected"

interface = gr.Interface(
    fn=spam_recog,
    inputs=gr.Textbox(lines=3, placeholder="Enter your email text..."),
    outputs="label",
    title="Spam Recognition"
)

if __name__ == "__main__":
    interface.launch()
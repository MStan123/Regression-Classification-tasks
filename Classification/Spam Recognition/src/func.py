from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def evaluate_model(true, predicted):
    accuracy = accuracy_score(true, predicted)
    recall = recall_score(true, predicted)
    precision = precision_score(true, predicted)
    f1 = f1_score(true, predicted)
    roc_auc = roc_auc_score(true, predicted)
    return accuracy, f1, precision, recall, roc_auc

def TSNE_plot(model_clf, data, labels, title):
    transformed_data = model_clf.decision_function(data)
    transformed_data = transformed_data.reshape(-1, 1)

    tsne_model = TSNE(n_components=2, random_state=0, init='random', perplexity=30, n_iter=300)
    tsne_data = tsne_model.fit_transform(transformed_data)

    tsne_data = np.vstack((tsne_data.T, labels)).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

    sns.scatterplot(data=tsne_df, x='Dim_1', y='Dim_2', hue='label', palette="bright")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

def report(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_m = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")

    print(f"Classification Report:")
    print(report)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_m, display_labels=classifier.classes_)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax)
    plt.title("Confusion Matrix")

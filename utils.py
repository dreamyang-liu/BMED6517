import umap
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from umap import plot
def get_umap_embedding(data, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
    reducer.fit(data)
    return reducer

def plot_umap_embedding(data, labels, n_neighbors=10, min_dist=0.2, n_components=2, random_state=42):
    reducer = get_umap_embedding(data, n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
    plt.figure(figsize=(10, 10))
    plot.points(reducer, labels=labels, theme='fire')
    plt.savefig('umap.png')

def plot_roc(y_test, y_pred_proba):
    fpr1, tpr1, _ = metrics.roc_curve(y_test,  y_pred_proba[:, 1])
    fpr0, tpr0, _ = metrics.roc_curve(1-y_test,  y_pred_proba[:, 0])
    plt.plot(fpr0,tpr0)
    plt.plot(fpr1,tpr1)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(['NORMAL', 'PNEUMONIA'])
    plt.savefig('roc1.png')
    print(metrics.auc(fpr0, tpr0))
    print(metrics.auc(fpr1, tpr1))

if __name__ == '__main__':
    pass

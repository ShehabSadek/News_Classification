import matplotlib.pyplot as plt
import numpy as np
def plot_accuracy_graph(model_names, accuracies):
    fig, ax = plt.subplots()

    x_pos = range(len(model_names))

    ax.bar(x_pos, accuracies, align='center')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right')

    ax.set_ylabel('Accuracy')

    ax.set_title('Model Accuracies')

    plt.tight_layout()
    plt.figure(figsize=(3, 3))
    plt.show()

def metrics(acc_values,pre_values,rec_values,f1_values):
    classifiers = ['LogisticRegression', 'MultinomialNB', 'RandomForestClassifier', 'SVC']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    metrics_values = np.array([acc_values, pre_values, rec_values, f1_values])

    bar_width = 0.2

    index = np.arange(len(classifiers))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i in range(len(metrics)):
        plt.bar(index + i * bar_width, metrics_values[i], bar_width, label=metrics[i], color=colors[i])

    plt.xlabel('Classifiers')
    plt.ylabel('Metric Value')
    plt.title('Performance Metrics for Classifiers')
    plt.xticks(index + bar_width * 1.5, classifiers)
    plt.legend()

    plt.show()

import matplotlib.pyplot as plt

def plot_accuracy_graph(model_names, accuracies):
    fig, ax = plt.subplots()

    x_pos = range(len(model_names))

    ax.bar(x_pos, accuracies, align='center')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right')

    ax.set_ylabel('Accuracy')

    ax.set_title('Model Accuracies')

    plt.tight_layout()
    plt.show()

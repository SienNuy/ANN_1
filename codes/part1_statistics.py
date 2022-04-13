import matplotlib.pyplot as plt

def plot_trainval_loss(train, val):
    plt.plot(train, 'ro--', linewidth=2, markersize=12, label='Train Loss')
    plt.plot(val, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12, label='Validatin Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.show()



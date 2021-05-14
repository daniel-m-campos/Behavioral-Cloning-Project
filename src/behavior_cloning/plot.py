def plot(history):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Please install matplotlib to use this function")
        return
    fig, ax = plt.subplots(1, 1)
    epochs = list(range(len(history["loss"])))
    ax.plot(epochs, history["loss"], label="Training set")
    ax.plot(epochs, history["val_loss"], label="Validation set")
    ax.set_title("Model Performance per Epoch")
    ax.set_ylabel("Mean Squared Error Loss")
    ax.set_xlabel("epoch")
    ax.legend(loc="upper right")
    return fig


if __name__ == "__main__":
    import pickle
    import pathlib

    path = pathlib.Path("../../data/track1")
    history = pickle.load(open(path / "history.pickle", "rb"))
    fig = plot(history)
    fig.savefig(path / "final_history.png")

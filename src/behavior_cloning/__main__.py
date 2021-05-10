from pathlib import Path

from data import read, convert
from model import create


def main(path: Path):
    X_train, y_train = convert(read(path))
    model = create()
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True)
    model.save(path / "model.h5")


if __name__ == "__main__":
    main(Path("/Users/danielmcampos/Downloads/data"))

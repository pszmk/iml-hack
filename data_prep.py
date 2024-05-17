from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple
from numpy import ndarray


RAW_DATA_DIR: Path = Path(__file__).parent / "data" / "raw"
PROCESSED_DATA_DIR: Path = Path(__file__).parent / "data" / "processed"
TRAIN_DATA_FILENAME: str = "training_data.csv"


def get_label_encoder():
    le = LabelEncoder()
    le.fit([-1, 0, 1])

    return le


def load_train(perform: bool = False, group: bool = False) -> tuple:
    data = pd.read_csv(RAW_DATA_DIR / TRAIN_DATA_FILENAME, sep=";").replace(
        ",", ".", regex=True
    )

    le = get_label_encoder()
    class_labels = le.transform(data["Class"].values.astype(np.int32))

    if perform:
        perform_variable = data["Perform"].values.astype(np.float32)

    if group:
        group_variable = pd.get_dummies(data[["Group"]], drop_first=True).values.astype(
            np.float32
        )
        data = data.drop(["Class", "Perform", "Group"], axis=1).values.astype(
            np.float32
        )
        data = np.concatenate([data, group_variable], axis=1)
    else:
        data = data.drop(["Class", "Perform", "Group"], axis=1).values.astype(
            np.float32
        )

    if perform:
        return data, class_labels, perform_variable
    else:
        return data, class_labels


def get_dataloader(
    data: np.ndarray,
    labels: np.ndarray,
    perform: np.ndarray = None,
    batch_size: int = 32,
) -> DataLoader:
    if perform is not None:
        dataset = TensorDataset(
            torch.tensor(data), torch.tensor(labels), torch.tensor(perform)
        )
    else:
        dataset = TensorDataset(torch.tensor(data), torch.tensor(labels))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def prep_crossvalidation(
    data: np.ndarray,
    labels: np.ndarray,
    perform: np.ndarray = None,
    batch_size: int = 32,
    n_splits: int = 5,
    seed: int = 42,
    dataloader: bool = True,
) -> List[Tuple[DataLoader, Tuple[ndarray]]]:
    np.random.seed(seed)
    indices = np.random.permutation(data.shape[0])
    data = data[indices]
    labels = labels[indices]
    if perform is not None:
        perform = perform[indices]

    fold_size = data.shape[0] // n_splits
    folds = []
    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size
        if i == n_splits - 1:
            end = data.shape[0]

        train_data = np.concatenate([data[:start], data[end:]])
        train_labels = np.concatenate([labels[:start], labels[end:]])
        val_data = data[start:end]
        val_labels = labels[start:end]

        if perform is not None:
            train_perform = np.concatenate([perform[:start], perform[end:]])
            val_perform = perform[start:end]

            folds.append(
                (
                    (
                        get_dataloader(
                            train_data, train_labels, train_perform, batch_size
                        )
                        if dataloader
                        else (
                            torch.tensor(train_data),
                            torch.tensor(train_labels),
                            torch.tensor(train_perform),
                        )
                    ),
                    (
                        torch.tensor(val_data),
                        torch.tensor(val_labels),
                        torch.tensor(val_perform),
                    ),
                )
            )
        else:
            folds.append(
                (
                    (
                        get_dataloader(train_data, train_labels, batch_size)
                        if dataloader
                        else (torch.tensor(train_data), torch.tensor(train_labels))
                    ),
                    (torch.tensor(val_data), torch.tensor(val_labels)),
                )
            )

    return folds

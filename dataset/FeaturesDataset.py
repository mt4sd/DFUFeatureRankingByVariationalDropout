from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

class FeaturesDataset(Dataset):
    def __init__(self, dataset_dir: str, normalize: bool = True):
        super(FeaturesDataset, self).__init__()
        X_df = pd.read_csv(os.path.join(dataset_dir, 'dfu_features_dataset_selected.csv'), index_col=0)
        y_df = pd.read_csv(os.path.join(dataset_dir, 'dfu_labels_dataset.csv'), index_col=0)
        
        self.features = X_df.columns.to_list()
        self.X = X_df.to_numpy().astype(np.float32)
        self.y = y_df.to_numpy().ravel()

        if normalize:
            self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


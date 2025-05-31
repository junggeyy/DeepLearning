import os
import torch
import pandas as pd
import numpy as np
import lightning as L
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataset import random_split

class AmesHousingDataset(Dataset):
    def __init__(self, csv_path, transform=None):

        columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
                    'Total Bsmt SF', 'SalePrice']
        
        df = pd.read_csv(csv_path, sep='\t', usecols=columns)
        df = df.dropna(axis=0)

        X = df[['Overall Qual',
                'Gr Liv Area',
                'Total Bsmt SF']].values
        y = df['SalePrice'].values

        sc_x = StandardScaler()
        sc_y = StandardScaler()
        X_std = sc_x.fit_transform(X)
        y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
        
        self.features = torch.tensor(X_std, dtype=torch.float)
        self.label = torch.tensor(y_std, dtype=torch.float).unsqueeze(1)

    def __getitem__(self, idx):
        return self.features[idx], self.label[idx]
    
    def __len__(self):
        return len(self.label)

class AmesHousingDataModule(L.LightningDataModule):
    def __init__(self,
                 csv_path='http://jse.amstat.org/v19n3/decock/AmesHousing.txt',
                 batch_size=32):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size

    def prepare_data(self):
        if not os.path.exists('AmesHousing.txt'):
            df = pd.read_csv(self.csv_path)
            df.to_csv('AmesHousing.txt', index=None)

    def setup(self, stage: str):
        all_data = AmesHousingDataset(csv_path='AmesHousing.txt')
        temp, self.val = random_split(all_data, [2500, 429],
                                      torch.Generator().manual_seed(1))
        self.train, self.test = random_split(temp, [2000, 500],
                                             torch.Generator().manual_seed(1))

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size,
            shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)
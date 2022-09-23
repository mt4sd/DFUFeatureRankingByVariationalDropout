import os
from .utils import DFUData
import pandas as pd

class INAOEDataset():
    def __init__(self, root_path, dm_group = False):
        cg_path = os.path.join(root_path, 'Control Group')
        dm_path = os.path.join(root_path, 'DM Group')

        self.dm_group = dm_group

        self.cg_subjects = sorted(list(map(lambda x: os.path.join(cg_path, x), os.listdir(cg_path))))
        self.dm_subjects = sorted(list(map(lambda x: os.path.join(dm_path, x), os.listdir(dm_path))))

    def set_group(self, dm_group):
        self.dm_group = dm_group

    def __len__(self):
        if self.dm_group:
            return len(self.dm_subjects)
        
        return len(self.cg_subjects) 

    def get_subjects_names(self):
        dataset = self.dm_subjects if self.dm_group else self.cg_subjects
        return list(map(lambda x: os.path.basename(x), dataset))

    def get_subject(self, idx, data:DFUData):
        dataset = self.dm_subjects if self.dm_group else self.cg_subjects

        subject = os.path.basename(dataset[idx])
        path = os.path.join(dataset[idx], 'Angiosoms') if data.value > 1 else dataset[idx]
        return pd.read_csv(os.path.join(path, subject + '_' + data.name + '.csv'), header=None).astype('float32')

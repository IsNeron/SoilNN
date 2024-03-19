import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

def generate_cfs_array_test(src_path: Path) -> np.array:
    filepath = {}
    for src in src_path.glob('**/*'):
            filepath.update({src.name: src})

    cfs_14 = []
    cfs_16 = []
    cfs_22 = []
    cfs_24 = []
    cfs_30 = []
    cfs_34 = []

    for name, path in filepath.items():
            function = np.genfromtxt(path, delimiter=',')
            match name[:2]:
                    case '14':
                        cfs_14.append(function)
                    case '16':
                        cfs_16.append(function)
                    case '22':
                        cfs_22.append(function)
                    case '24':
                        cfs_24.append(function)
                    case '30':
                        cfs_30.append(function)
                    case '34':
                        cfs_34.append(function)

    return np.array([cfs_14, cfs_16, cfs_22, cfs_24, cfs_30, cfs_34])


def prepare_permeability(src_path: Path) -> np.array:
    result = []
    for src in src_path.glob('**/*'):
            result.append(np.genfromtxt(src, delimiter=','))
    
    return np.array(result)



class ExperimentalDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return len(self.data)
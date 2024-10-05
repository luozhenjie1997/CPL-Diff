from torch.utils.data import Dataset


# 返回数据和单标签
class XYDataset(Dataset):
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        self.size = len(dataFrame)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {'sequences': self.dataFrame['sequence'].iloc[idx],
                'labels': self.dataFrame['label'].iloc[idx]}


class XDataset(Dataset):
    """
    返回数据，输入的数据集为DataFrame

    """
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        self.size = len(dataFrame)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {'sequences': self.dataFrame['sequence'].iloc[idx]}

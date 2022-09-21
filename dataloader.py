import torch

class HDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x1 = x[self.y == 1]
        self.x0 = x[self.y == 0]
        self.y1 = y[self.y == 1]
        self.y0 = y[self.y == 0]
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class HDataLoader(object):
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def __iter__(self):
        for data in self.data_loader:
            yield data

    def __len__(self):
        return len(self.data_loader)

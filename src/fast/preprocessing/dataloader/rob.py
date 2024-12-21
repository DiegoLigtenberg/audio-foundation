from datetime import datetime
from time import sleep

import torch
from torch.utils.data import DataLoader, Dataset


def current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.arange(1024)

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        print(f"{current_time()} - loading idx {idx:02} - worker {worker_info.id}")
        sleep(1)
        print(f"{current_time()} - #LOADED idx {idx:02} - worker {worker_info.id}")
        x = self.data[idx]
        return x

    def __len__(self):
        return len(self.data)


def main():
    data = DataLoader(
        dataset=MyDataset(),
        batch_size=8,
        num_workers=4,
        shuffle=False,
    )
    for batch in data:
        print(f"{current_time()} - using batch: {batch}")
        sleep(2)


if __name__ == "__main__":
    main()
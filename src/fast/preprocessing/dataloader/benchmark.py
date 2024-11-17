from pathlib import Path
from time import time

import torch


def save_data(save_path, nr_sec: float, token_sec: float, nr_songs: int = 100):
    save_path.mkdir(exist_ok=True, parents=True)

    for i in range(nr_songs):
        song = torch.rand(1024,int(nr_sec/token_sec),dtype=torch.float32)#   int(nr_sec / token_sec), dtype=torch.float32)
        torch.save(song, save_path / f"song_{i:03}.pt")


def load_data(save_path, nr_songs: int = 100):
    for i in range(nr_songs):
        song = torch.load(save_path / f"song_{i:03}.pt")


if __name__ == "__main__":
    save_path = Path("benchmark/")
    for nr_sec in [20]: #, 30, 60, 120, 180, 240, 300]:
        # nr_sec = 10
        token_sec = 0.0390625 # 0.02
        nr_songs = 1000
        save_data(save_path, nr_sec=nr_sec, token_sec=token_sec, nr_songs=nr_songs)
        print(
            f"song segment length: {nr_sec} sec, token resolution: {token_sec} sec",
            end=", ",
            flush=True,
        )

        start = time()
        load_data(save_path, nr_songs=nr_songs)
        end_time = time() - start

        print(f"loaded {nr_songs} songs in {end_time:0.2f} sec")
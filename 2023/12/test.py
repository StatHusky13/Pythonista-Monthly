import numpy as np
import pandas as pd
import random
from main import CellType, create_map, get_coverage, char_lookup
import time


def test():
    n_tests = 1000
    test_num = 0

    bin_step = 10 ** 5
    bins = list(range(1, 1000**2, bin_step)) # n cells
    bins.append(float("inf"))
    data = [] # runtime
    
    time_avg: float = 0
    while test_num <= n_tests:
        shape = (random.randint(1, 10*3), random.randint(1, 10**3))
        n_sensors, n_tags = (
            random.randint(0, shape [0] // 2),
            random.randint(0, shape[1] // 2),
        )
        grid = distribute_cells(shape, n_sensors, n_tags)

        start = time.time()

        try:
            print(grid.shape)
            get_coverage(grid)
        except Exception as e:
            print(
                f"Exception\nSize: {shape}\nS, T: {(n_sensors, n_tags)}\nTime: {(time.time()-start) * 1000:.3f}ms"
            )
            raise e

        time_avg = (time_avg * test_num + (time.time() - start)) / (test_num + 1)
        data.append({
            "Runtime [ms]": (time.time()-start) * 1000, 
            "N_Cells": shape[0]*shape[1]
        })
        test_num += 1

        if test_num % 100 == 0:
            print(f"{test_num} / {n_tests}")
    
    df = pd.DataFrame(data)
    df["Cell_Group"] = pd.cut(df["N_Cells"], bins=bins, labels=[f"{k} to {k+bin_step}" for k in bins[:-1]])
    grouped_df = df.groupby("Cell_Group", observed=True)['Runtime [ms]'].mean().reset_index()
    print(grouped_df)


def distribute_cells(shape: tuple[int, int], n_sensors: int, n_tags: int):
    base = np.full(shape, CellType.EMPTY.value)
    distr(base, CellType.SENSOR.value, n_sensors)
    distr(base, CellType.TAG.value, n_tags)
    return base


def distr(grid: np.ndarray, value: int, n_values: int):
    n_rows, n_cols = grid.shape
    n_runs = 10**5 + n_values

    while n_values >= 1 and n_runs >= 1:
        cell = int(random.random() * n_rows), int(random.random() * n_cols)
        if grid[cell] == CellType.EMPTY.value:
            grid[cell] = value
            n_values -= 1
        n_runs -= 1

    return grid

if __name__ == "__main__":
    test()
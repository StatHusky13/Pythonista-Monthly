import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from main import CellType, get_coverage
import time


def test():
    n_tests = int(1e4)
    test_num = 0

    bin_step: int = int(10e3 / 4)
    n_bins = 10
    m_dim = int((bin_step * n_bins) ** 0.5)
    bins = list(range(1, bin_step * n_bins, bin_step)) # n cells
    

    bins.append(float("inf"))
    data = []
    
    while test_num <= n_tests:
        shape = (random.randint(1, m_dim), random.randint(1, m_dim))
        n_sensors, n_tags = (
            random.randint(0, shape [0] // 10),
            random.randint(0, shape[1] // 10),
        )
        grid = distribute_cells(shape, n_sensors, n_tags)

        start = time.time()

        try:
            get_coverage(grid)
        except Exception as e:
            print(
                f"Exception\nSize: {shape}\nS, T: {(n_sensors, n_tags)}\nTime: {(time.time()-start) * 1000:.3f}ms"
            )
            raise e

        data.append({
            "Runtime [ms]": (time.time()-start) * 1000, 
            "N": shape[0]*shape[1]
        })
        test_num += 1

        if test_num % 100 == 0:
            print(f"{test_num} / {n_tests}")
    
    df = pd.DataFrame(data)
    df["N_Cells"] = pd.cut(df["N"], bins=bins, labels=[f"{k} to {k+bin_step}" for k in bins[:-1]])
    grouped_df = df.groupby("N_Cells", observed=True).agg({"Runtime [ms]": "mean", "N": "size"}).reset_index()
    grouped_df.rename(columns={'N': 'N_Bin'}, inplace=True)

    print(grouped_df)
    print(bins[:-1], list(grouped_df["Runtime [ms]"]))
    model = np.poly1d(np.polyfit(bins[:-1], list(grouped_df["Runtime [ms]"]), 2))
    plt.scatter(bins[:-1], grouped_df["Runtime [ms]"])

    polyline = np.linspace(0, bin_step*n_bins, 10)
    plt.plot(polyline, model(polyline))
    plt.show()


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
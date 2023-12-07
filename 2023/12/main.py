"""
See:  2023/12/readme.md
"""

import numpy as np
from collections import deque
from enum import Enum


class CellType(Enum):
    EMPTY = 0
    SENSOR = 1
    TAG = 2


char_lookup: dict[str, int] = {
    ".": CellType.EMPTY.value,
    "S": CellType.SENSOR.value,
    "T": CellType.TAG.value,
}


def create_map(original_map: str) -> list[list[int]]:
    # turn the grid into a grid of integers (see char lookup hashmap)
    grid = parse_map(original_map)

    # get the coverage mapping
    coverage = get_coverage(grid)

    return coverage.astype(np.int8).tolist()


def parse_map(map: str) -> np.ndarray:
    n_rows: int = map.count("\n") + 1 
    replaced = [char_lookup[char] for char in map.strip() if char != "\n"]
    n_cols: int = len(replaced) // n_rows

    return np.array(replaced, dtype=np.int8).reshape(n_rows, n_cols)


def get_coverage(grid: np.ndarray) -> np.ndarray:
    base_map = np.ones_like(grid, dtype=np.bool_)
    sensor_indicies = np.argwhere(grid == CellType.SENSOR.value)

    for sensor in sensor_indicies:
        sensor_coverage = bfs_map(sensor, grid)
        base_map = np.bitwise_and(base_map, sensor_coverage)

    return base_map


def bfs_map(start: tuple[int, int], grid: np.ndarray) -> np.ndarray:
    visited_grid = np.zeros_like(grid, dtype=np.bool_)

    max_row = grid.shape[0]
    max_col = grid.shape[1]

    s_row = start[0]
    s_col = start[1]

    queue = deque()
    queue.append((s_row, s_col))

    while queue:
        next: tuple[int, int] = queue.popleft()
        row, col = next[0], next[1]

        if grid[row, col] == CellType.TAG.value:
            base = np.ones_like(grid, dtype=np.bool_)
            t: int = manhattan((row, col), start)

            for scan_row in range(
                max(0, s_row - t), 
                min(s_row + t + 1, grid.shape[0])
            ):
                span = t - abs(s_row - scan_row)

                for c in range(
                    max(0, s_col - span), 
                    min(s_col + span + 1, grid.shape[1])
                ):
                    base[scan_row, c] = 0
            return base

        for n_row, n_col in (
            (row + 1, col),
            (row - 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if not ((0 <= n_row < max_row) and (0 <= n_col < max_col)):
                continue
            if visited_grid[n_row, n_col]:
                continue
            visited_grid[n_row, n_col] = True
            queue.append((n_row, n_col))

    return np.ones_like(grid)


def manhattan(p1: tuple[int, int], p2: tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


if __name__ == "__main__":
    file_input = open("./2023/12/inputs/2.txt", "r").read()
    result = create_map(file_input)

    print(np.array(result))
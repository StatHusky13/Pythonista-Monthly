import numpy as np
from collections import deque
from enum import Enum
from pprint import pprint


class GridIndex(Enum):
    EMPTY = 0
    SENSOR = 1
    TAG = 2


char_lookup: dict[str, int] = {
    ".": GridIndex.EMPTY.value,
    "S": GridIndex.SENSOR.value,
    "T": GridIndex.TAG.value,
}


def create_map(original_map: str) -> list[list[int]]:
    # approach:
    # get the indicies of each tag
    # for each sensor, calculate the taxicab distance to the nearest tag
    # then, redraw the map, if a square is within the radius of a sensor,
    # make it 1. if it isn't, make it 0.

    replaced = [char_lookup[char] for char in original_map.strip() if char != "\n"]

    n_rows = original_map.count("\n") + 1
    n_cols = len(replaced) // n_rows

    # Convert the list to a NumPy array and reshape
    grid = np.array(replaced, dtype=np.int8).reshape(n_rows, n_cols)

    sensor_indicies = np.argwhere(grid == GridIndex.SENSOR.value)

    # for each sesnor, find the closest tag using simple bfs
    base_map = np.ones_like(grid, dtype=np.bool_)
    for sensor in sensor_indicies:
        # print("sensor", sensor)
        # print(new.astype(np.int8))
        base_map = np.bitwise_and(base_map,  bfs_map(sensor, grid)) # places that need to be searched in BOTH maps (prev and new)
    
    return base_map.astype(np.int8).tolist()

def bfs_map(start: tuple[int, int], grid: np.ndarray) -> np.ndarray:
    """
    given a starting index and a grid of values, returns
    a new grid where any cell within the radius of a sensor
    is a 1. any cell outside the radius of a sensor is a 0.
    """
    visited_grid = np.zeros_like(grid, dtype=np.bool_)

    maxrow = grid.shape[0]
    maxcol = grid.shape[1]

    s_row = start[0]
    s_col = start[1]

    queue = deque()
    queue.append((s_row, s_col))

    # something to note is that even if we find the tag, we still need to
    # mark all other cells in the queue as a 0 because they are within the
    # radius of the sensor

    while queue:
        next: tuple[int, int] = queue.popleft()
        row, col = next[0], next[1]  # values to search
        visited_grid[row, col] = True

        if grid[row, col] == GridIndex.TAG.value:
            # print("TAG SEQ")
            # we found the tag. for every value that is equal to this distance from
            # start, set it to 0. This is to finish the circle around the start
            base = np.ones_like(grid, dtype=np.bool_)
            t = manhattan((row, col), start) + 1 # lol apparently this does it bc "distance in inclusive idk"
            # print(f"tag [{row} {col}] ({t})")

            for scan_row in range(
                max(0, s_row - t), 
                min(s_row + t + 1, grid.shape[0])
            ):
                span = (t - abs(s_row - scan_row)) # distance from center column to each side of the row discovered on each side
                # print(" ", end="")
                for c in range(
                    max(0, s_col - span), 
                    min(s_col + span + 1, grid.shape[1])
                ):
                    # print(c, end=" ")
                    base[scan_row, c] = 0
                # print()
            return base

        for n_row, n_col in (
            (row + 1, col),
            (row - 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if not ((0 <= n_row < maxrow) and (0 <= n_col < maxcol)):
                continue
            if visited_grid[n_row, n_col] == True:
                continue
            queue.append((n_row, n_col))

    return np.ones_like(grid) # base base - all need searching


def manhattan(p1: tuple[int, int], p2: tuple[int, int]):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

if __name__ == "__main__":
    file_input = open("2023_12_input.txt", "r").read()
    result = create_map(file_input)

    print(np.array(result)) # nparray print is better lol

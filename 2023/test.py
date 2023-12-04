import numpy as np

def points_at_distance_numpy(center_row, center_col, T):
    rows = np.arange(center_row - T, center_row + T + 1)
    cols = center_col + T - np.abs(rows - center_row)

    result_points = np.column_stack((np.repeat(rows, 2 * T + 1), np.tile(cols, 2 * T + 1)))

    return result_points

# Example usage:
center_row, center_col = 2, 2
distance_T = 2

result = points_at_distance_numpy(center_row, center_col, distance_T)
print(result)
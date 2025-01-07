import numpy as np
def rotate(input_grid):
  return np.rot90(input_grid)
def rotate180(input_grid):
  return np.rot180(input_grid)
def rotate270(input_grid):
  retunr np.rot270(input_grid)
def flip_horizontal(input_grid):
  return np.fliphr(input_grid)
def flip_vertical(input_grid):
  return np.flipup(input_grid)

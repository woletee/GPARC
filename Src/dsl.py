import numpy as np
#Note
"List of DSLs used "
def diamirror(input_grid):
    return np.transpose(input_grid)
def Hmirror(input_grid: np.ndarray) -> np.ndarray:
    return np.fliplr(input_grid)
def Vmirrors(input_grid: np.ndarray) -> np.ndarray:
    return np.flipud(input_grid)
def transform_blue_to_red(input_grid):
    grid = np.array(input_grid)
    # Replace all blue colored pixles with red
    output_grid = np.where(grid == 1, 2, grid)
    return output_grid

def upscale(input_grid, upscale_factor=3):
    def expand_pixel_with_grid(pixel, input_grid):
        if pixel == 0:
            return np.zeros((upscale_factor, upscale_factor), dtype=int)
        else:
            return np.full((upscale_factor, upscale_factor), pixel, dtype=int)
    input_rows, input_cols = input_grid.shape
    output_grid = np.zeros((input_rows * upscale_factor, input_cols * upscale_factor), dtype=int)
    for r in range(input_rows):
        for c in range(input_cols):
            expanded_block = expand_pixel_with_grid(input_grid[r, c], input_grid)
            output_grid[r * upscale_factor: (r + 1) * upscale_factor, c * upscale_factor: (c + 1) * upscale_factor] = expanded_block
    return output_grid

def flip_horizontal(grid: np.ndarray) -> np.ndarray:
    return np.fliplr(grid)

def flip_vertical(grid: np.ndarray) -> np.ndarray:
    return np.flipud(grid)

def rotate_90(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=-1)

def rotate_180(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=2)

def rotate_270(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=1)

def identity(grid: np.ndarray) -> np.ndarray:
    return grid
#we need to finish settling up and changing the dsls into the numpy array 

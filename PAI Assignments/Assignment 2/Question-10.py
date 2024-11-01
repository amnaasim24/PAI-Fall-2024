
import numpy as np

def normalize_array(arr):
    mean = np.mean(arr)
    std_dev = np.std(arr)
    normalized = (arr - mean) / std_dev
    return normalized

sample_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
normalized_array = normalize_array(sample_array)

print("Original Array:\n", sample_array)
print("Normalized Array:\n", normalized_array)

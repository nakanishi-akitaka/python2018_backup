# r^2 based on the latest measured y-values
import numpy as np


# Calculate r^2 based on the latest measured y-values
# measured_y and estimated_y must be vectors.
def r2lm(measured_y, estimated_y):
    measured_y = np.array(measured_y).flatten()
    estimated_y = np.array(estimated_y).flatten()
    return float(1 - sum((measured_y - estimated_y) ** 2) / sum((measured_y[1:] - measured_y[:-1]) ** 2))

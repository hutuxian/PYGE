import numpy as np
path="./data/"

# test data
np.random.random_sample((1, 1, 28, 28)).astype(np.float16).tofile(path + "conv2d_variable_input_x.bin")
np.random.random_sample((1, 10)).astype(np.float16).tofile(path + "labels_variable_input.bin")



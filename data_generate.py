import numpy as np
path = "./data/"

a = np.random.rand(2, 1).astype(np.float16).tofile(path + "ge_variable_input_a.bin")

b = np.random.rand(2, 1).astype(np.float16).tofile(path + "ge_variable_input_b.bin")
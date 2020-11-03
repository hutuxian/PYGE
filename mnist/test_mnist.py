"""
-*- coding:utf-8 -*-
"""
import numpy as np
import unittest
import ge
from mnist import PyGe

PATH = "./data/"

def check_ret(message, ret):
    """
    check return value
    """
    if ret != 0:
        raise Exception("{} failed ret = {}".format(message, ret))

def get_class_methods(class_name):
    method_list = []
    for method in dir(class_name):
        str_list = method.split('_')
        if str_list[0] == 'test':
            method_list.append(str_list)

    method_list = sorted(method_list, key=lambda x: x[2])

    methods = []
    for method in method_list:
        methods.append("_".join(method))

    return methods

def get_test_case_by_list(suite, case_class, methods, list):
    for case_no in list:
        for method in methods:
            if method.find(case_no) >= 0:
                suite.addTest(case_class(method))
    return suite

def get_test_case_by_opt(suite, case_class, methods, opt):
    for method in methods:
        if method.find(opt) >= 0:
            suite.addTest(case_class(method))

    return suite

def switch_cases(case_class, opt):
    """
    :param case_class:
    :param opt:
        "all"   : all test cases
        "001"   : test case number 001
        "err"   : all error test cases
    :return:
    """
    suite = unittest.TestSuite()
    methods = get_class_methods(case_class)

    if type(opt) == list:
        return get_test_case_by_list(suite, case_class, methods, opt)

    if opt == "all":
        for method in methods:
            suite.addTest(case_class(method))
        return suite
    if opt == "err":
        suite = get_test_case_by_opt(suite, case_class, methods, opt)
    else:
        for method in methods:
            if method.find(opt) >= 0:
                suite.addTest(case_class(method))
                break
    return suite


class TestGe(unittest.TestCase):
    def test_001_mnist(self):
        """
        network mnist by GE
        |============= conv ============== | ===== pool ===== | ===== conv ===== | ===== pool ===== | ===== fc ===== | === softmax === |
        |input: 28×28×1 -> cells: 28×28×32 |  cells: 14×14×32 |  cells: 14×14×64 |  cells:  7×7×64  |   1×1×1024     |     1×1×10      |
        |  Conv     : 5×5                  |   Pool   : 2×2   |    conv   : 5×5  |   Pool   : 2×2   |                |                 |
        |  filters  : 32                   |   stride : 2     |    filter :64    |   stride : 2     |   dropout 0.5  |                 |
        |  padding  : 2                    |                  |    padding:2     |                  |                |                 |
        ================================================================================================================================
        """
        config = {"ge.exec.deviceId": "0", "ge.graphRunMode": "1", "ge.exec.precision_mode": "allow_mix_precision"}
        options = {}
        ge_handle =PyGe(config, options)

        inputs = []
        input_shapes = [1, 28, 28, 1]
        input_path = PATH + "conv2d_variable_input_x.bin"
        input_tensor = ge_handle.get_tensor_from_bin(input_path, input_shapes, fmt=ge.FORMAT_NHWC, dt=ge.DT_FLOAT16)
        inputs.append(input_tensor)
        input_shapes = [1, 10]
        input_path = PATH + "labels_variable_input.bin"
        in_labels_tensor = ge_handle.get_tensor_from_bin(input_path, input_shapes, fmt=ge.FORMAT_NCHW, dt=ge.DT_FLOAT16)
        inputs.append(in_labels_tensor)

        ge_handle.init_conv2d_param(32, [1, 28, 28, 1], [32, 5, 5, 1])
        ge_handle.init_conv2d_param(64, [1, 14, 14, 32], [64, 5, 5, 32])
        ge_handle.init_fc_param(1024, [3136, 1024], [1024,])
        ge_handle.init_fc_param(10, [1024, 10], [10,])
        ge_handle.init_pool_param()

        # init graph
        print("++++++ init mnist ++++++")
        graph = ge.Graph("mnist")
        var_name, var_value, var_desc = ge_handle.get_var_parameters()
        ge_handle.init_graph_param(graph, var_desc, var_name, var_value)

        # mnist forward graph
        print("++++++ forward ++++++")
        ge_handle.mnist_forward(graph)
        ge_handle.add_graph(1, graph)
        # run graph
        outputs_forward = ge_handle.run_graph(1, inputs)
        ge_handle.update_input_params(inputs, outputs_forward)
        # mnist backward graph
        print("++++++ backward ++++++")
        graph_back = ge.Graph("mnist_back")
        ge_handle.mnist_backprop(graph_back)
        ge_handle.add_graph(2, graph_back)
        outputs_back = ge_handle.run_graph(2, [outputs_forward[0], outputs_forward[1]])
        ge_handle.update_net_params(outputs_back)
        ge_handle.print([], outputs_back, np.float16)

if __name__ == "__main__":
    suite = switch_cases(TestGe, ["001"])
    unittest.TextTestRunner(verbosity=2).run(suite)

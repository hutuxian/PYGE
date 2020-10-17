"""
-*- coding:utf-8 -*-
"""
import numpy as np
import unittest
import ge

PATH = "./data/"
var_name = ['x1', 'x2']

def check_ret(message, ret):
    """
    check return value
    """
    if ret != 0:
        raise Exception("{} failed ret = {}".format(message, ret))


class PyGe(object):
    """
    class PyGe
    function: encapsulating methods of GE
    """
    def __init__(self, config, options):
        ret = ge.ge_initialize(config)
        check_ret("ge_initialize", ret)
        print("Initialize ge success.")

        self.session = ge.Session(options)
        if self.session:
            print("Create session success.")
        else:
            print("Create session fail.")
        
    def __del__(self):
        ret = ge.ge_finalize()
        check_ret("ge_finalize", ret)
        print("Finalize ge success.")
        del self.session
    
    def get_data_type_size(self, dt):
        """
        get data type size
        """
        dailation = 1
        if dt == ge.DT_FLOAT:
            dailation = 4
        elif dt == ge.DT_FLOAT16:
            dailation = 2
        elif dt == ge.DT_INT16:
            dailation = 2
        elif dt == ge.DT_UINT16:
            dailation = 2
        elif dt == ge.DT_INT32:
            dailation = 4
        elif dt == ge.DT_UINT32:
            dailation = 4
        elif dt == ge.DT_INT64:
            dailation = 8
        elif dt == ge.DT_UINT64:
            dailation = 8
        elif dt == ge.DT_INT8:
            dailation = 1
        return dailation

    def get_tensor_data(self, tensor, dtype):
        """
        transform data
        """
        data = np.array(tensor.get_data(), dtype=np.uint8)
        b_arr = data.tobytes()
        arr_2 = np.frombuffer(b_arr, dtype=dtype)
        return arr_2
    
    def get_tensor_from_bin(self, in_path, shape_list, dformat=ge.FORMAT_ND, data_type=ge.DT_FLOAT16):
        """
        read bin to generate input data
        """
        size = 1
        for i in range(len(shape_list)):
            size *= shape_list[i]
        data_len = size * self.get_data_type_size(data_type)

        np_in = np.fromfile(in_path, dtype=np.uint8)
        np_size = np_in.size * np_in.itemsize
        assert np_size == data_len

        input_tensor_desc = ge.TensorDesc(ge.Shape(shape_list), dformat, data_type)
        input_tensor_desc.set_real_dim_cnt(len(shape_list))
        input_tensor = ge.Tensor(input_tensor_desc, np_in)
        return input_tensor
    
    def gen_tensor(self, tensor_shape, value, data_type):
        """
        generate tensor
        """
        size = 1
        for i in range(len(tensor_shape)):
            size *= tensor_shape[i]
        
        np_data = np.zeros(size, dtype=np.float16)
        for i in range(size):
            np_data[i] = value
        np_data = np.frombuffer(np_data.tobytes(), dtype=np.uint8)

        input_tensor_desc = ge.TensorDesc(ge.Shape(tensor_shape), ge.FORMAT_ND, data_type)
        tensor = ge.Tensor(input_tensor_desc, np_data)

        return tensor

    def add_graph(self, graph_id, graph):
        """
        add graph
        """
        ret = self.session.add_graph(graph_id, graph)
        check_ret("add_graph", ret)
        print("Session add {} success.".format(graph_id))
    
    def run_graph(self, graph_id, in_tensor):
        """
        run graph
        """
        out_tensor, ret = self.session.run_graph(graph_id, in_tensor)
        check_ret("run_graph", ret)
        print("Session run {} success.".format(graph_id))
        return out_tensor


def test_op_factory():
    """
    main
    """
    config = {"ge.exec.deviceId": "0", "ge.graphRunMode": "1"}
    options = {"a": "b", "ge.trainFlag": "1"}
    ge_handle =PyGe(config, options)

    graph = ge.Graph("Add")

    data_x1_shape = ge.OperatorFactory.create_operator("x1", "Data").set_attr_int32("index", 0)
    data_x2_shape = ge.OperatorFactory.create_operator("x2", "Data").set_attr_int32("index", 1)

    graph.add_op(data_x1_shape)
    graph.add_op(data_x2_shape)

    add = ge.OperatorFactory.create_operator("add", "Add").set_input("x1", data_x1_shape).set_input("x2", data_x2_shape)
    
    in_operator = [data_x1_shape, data_x2_shape]
    out_operator = [add]
    graph.set_inputs(in_operator).set_outputs(out_operator)
    graph.add_op(add)

    # add graph
    ge_handle.add_graph(0, graph)

    # run graph
    input_add = []
    input_shapes = [2, 1]
    input_a_path = PATH + "ge_variable_input_a.bin"
    input_b_path = PATH + "ge_variable_input_b.bin"
    input_tensor_a = ge_handle.get_tensor_from_bin(input_a_path, input_shapes)
    input_add.append(input_tensor_a)
    input_tensor_b = ge_handle.get_tensor_from_bin(input_b_path, input_shapes)
    input_add.append(input_tensor_b)
    output_add = ge_handle.run_graph(0, input_add)

    print('a=', ge_handle.get_tensor_data(input_add[0], np.float16),
        '\nb=', ge_handle.get_tensor_data(input_add[1], np.float16),
        '\nout=', ge_handle.get_tensor_data(output_add[0], np.float16))


def test_random():
    config = {"ge.exec.deviceId": "0", "ge.graphRunMode": "1"}
    options = {}
    ge_handle = PyGe(config, options)

    graph = ge.Graph("RandomUniform")

    # input
    size = 2
    np_data = np.array([2, 2]).astype(np.int32)
    np_data = np.frombuffer(np_data.tobytes(), dtype=np.uint8)
    input_tensor_desc = ge.TensorDesc(ge.Shape([2]), ge.FORMAT_NHWC, ge.DT_INT32)
    input_tensor = ge.Tensor(input_tensor_desc, np_data)
    input_random = []
    input_random.append(input_tensor)

    # 构图
    np_in = np.array([2, 2, 2, 2]).astype(np.int32)
    data_buffer = np.frombuffer(np_in.tobytes(), dtype=np.uint8)
    shape_tensor_desc = ge.TensorDesc(ge.Shape([4]), ge.FORMAT_NHWC, ge.DT_INT32)
    shape_tensor_desc.set_real_dim_cnt(1)
    shape_tensor = ge.Tensor(shape_tensor_desc, data_buffer)
    data1 = ge.OperatorFactory.create_operator("const", "Const").set_attr_tensor("value", shape_tensor)
    shape_const_desc = ge.TensorDesc(ge.Shape([4]), ge.FORMAT_NHWC, ge.DT_INT32)
    data1.update_output_desc("y", shape_const_desc)
    result_output = ge.OperatorFactory.create_operator("random", "RandomUniform").set_input(
        "shape", data1).set_attr_dtype("dtype", ge.DT_FLOAT)

    inputs = [data1]
    outputs = [result_output]

    graph.set_inputs(inputs).set_outputs(outputs)

    ge_handle.add_graph(0, graph)
    output_random = ge_handle.run_graph(0, input_random)

    print('out = ', ge_handle.get_tensor_data(output_random[0], np.float32))

    ge_handle.session.remove_graph(0)

if __name__ == '__main__':
    # test_op_factory()
    test_random()


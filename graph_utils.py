import numpy as np
import op
import ge

def get_data_type_size(dt):
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

def get_tensor_data(tensor, dtype):
    data = np.array(tensor.get_data(), dtype=np.uint8)
    b_arr = data.tostring()
    arr_2 = np.frombuffer(b_arr, dtype=dtype)
    return arr_2

# users can load data from data dir
def get_tensor_from_bin(in_path, shape_list, format=ge.FORMAT_ND, data_type=ge.DT_FLOAT16):
    size = 1
    for i in range(len(shape_list)):
        size *= shape_list[i]
    data_len = size * get_data_type_size(data_type)

    np_in = np.fromfile(in_path, dtype=np.uint8)
    np_size = np_in.size * np_in.itemsize
    assert np_size == data_len
    
    input_tensor_desc = ge.TensorDesc(ge.Shape(shape_list), format, data_type)
    input_tensor_desc.set_real_dim_cnt(len(shape_list))
    input_tensor = ge.Tensor(input_tensor_desc, np_in)

    return input_tensor

def gen_tensor(tensor_shape, value):
    size = 1
    for i in range(len(tensor_shape)):
        size *= tensor_shape[i]
    
    np_data = np.zeros(size, dtype=np.float16)
    for i in range(size):
        np_data[i] = value

    input_tensor_desc = ge.TensorDesc(ge.Shape(tensor_shape), ge.FORMAT_ND, ge.DT_FLOAT16)
    tensor = ge.Tensor()
    tensor.set_tensor_desc(input_tensor_desc)
    tensor.set_data(np_data)

    return tensor

#generate init graph
def gen_init_graph(graph_name, tensor_desc_list, var_name, var_values):
    graph = ge.Graph(graph_name)
    in_operator = []
    out_operator = []
    for i in range(len(tensor_desc_list)):
        tensor_desc_list[i].set_real_dim_cnt(tensor_desc_list[i].get_shape().get_dim_num())
        tensor = gen_tensor(tensor_desc_list[i].get_shape().get_dims(), var_values[i])

        var_const = op.Constant().set_attr_value(tensor)
        var_const.update_output_desc_y(tensor_desc_list[i])

        var_init = op.Variable(var_name[i])
        var_init.update_output_desc_y(tensor_desc_list[i])
        var_assign = op.Assign().set_input_ref(var_init).set_input_value(var_const)
        in_operator.append(var_init)

    graph.set_inputs(in_operator).set_outputs(out_operator)
    return graph

#generate add graph
def gen_add_graph(graph_name, var_desc_list, var_name_list):
    graph = ge.Graph(graph_name)

    data_x1_shape = op.Data("x1").set_attr_index(0)
    data_x2_shape = op.Data("x2").set_attr_index(1)

    var_x1 = op.Variable(var_name_list[0])
    var_x2 = op.Variable(var_name_list[1])
    var_x1.update_output_desc_y(var_desc_list[0])
    var_x2.update_output_desc_y(var_desc_list[1])

    graph.add_op(var_x1)
    graph.add_op(var_x2)

    add = op.Add().set_input_x1(data_x1_shape).set_input_x2(data_x2_shape)

    in_operator = [data_x1_shape, data_x2_shape]
    out_operator = [add]
    graph.set_inputs(in_operator).set_outputs(out_operator)
    graph.add_op(add)

    return graph


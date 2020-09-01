import numpy as np
import ge 
import graph_utils

PATH = "./data/"
var_name = ['x1','x2']

def check_ret(message, ret):
    if ret != 0:
        raise Exception("{} failed ret={}".format(message, ret))

def test_graph():
    # 0. System init
    config = {
        "ge.exec.deviceId": "0",
        "ge.graphRunMode": "1"
    }
    ret = ge.ge_initialize(config)
    check_ret("ge_initialize", ret)
    print("liInitiaze ge success.")

    # 1. Generate graph
    desc = ge.TensorDesc(ge.Shape([2,1]), ge.FORMAT_ND, ge.DT_FLOAT16)
    var_tensor_desc = [desc, desc]

    # 1.1init graph
    init_graph_id = 0
    init_var_graph = graph_utils.gen_init_graph("InitVarGraph", var_tensor_desc, var_name, [0, 0])
    print("Generate init graph success.")

    # 1.2 add graph
    add_graph_id = 1
    add_graph = graph_utils.gen_add_graph("AddGraph", var_tensor_desc, var_name)
    print("Generate add graph success.")

    # 2. create session
    options = {
        "a": "b",
        "ge.trainFlag": "1"
    }
    session = ge.Session(options)
    if session:
        print("Create session success.")
    else:
        print("Create session fail.")
    
    # 3. add graph
    ret = session.add_graph(init_graph_id, init_var_graph)
    check_ret("AddGraph init_graph_id", ret)
    print("Session add init graph success.")
    ret = session.add_graph(add_graph_id, add_graph)
    check_ret("AddGraph add_graph_id", ret)
    print("Session add ADD graph success.")

    # 4. Run graph
    input_init, input_add = [], []
    output_init, output_add = [], []
    output_init, ret = session.run_graph(init_graph_id, input_init)
    check_ret("RunGraph init_graph_id", ret)
    print("Session run Init graph success.")

    input_shapes = [2, 1]
    input_a_path = PATH + "ge_variable_input_a.bin"
    input_b_path = PATH + "ge_variable_input_b.bin"
    input_tensor_a = graph_utils.get_tensor_from_bin(input_a_path, input_shapes)
    input_add.append(input_tensor_a)
    input_tensor_b = graph_utils.get_tensor_from_bin(input_b_path, input_shapes)
    input_add.append(input_tensor_b)

    output_add, ret = session.run_graph(add_graph_id, input_add)
    check_ret("RunGraph add_graph_id", ret)
    print("Session run add graph success.")

    print('a=', graph_utils.get_tensor_data(input_add[0], np.float16),
        '\nb=', graph_utils.get_tensor_data(input_add[0], np.float16),
        '\nout=', graph_utils.get_tensor_data(output_add[0], np.float16))

    # 5.Optional operation: If a graph is runned before, and want to run again,
    # you need to check whether graph needed to rebuild,
    # so a graph is runned before, and needed to rebuild, user should remove it from GE first,
    # then add graph again and rebuild it.
    # 6. system finalize
    ret = ge.ge_finalize()
    check_ret("ge_finalize", ret)
    print("Finalize ge success.")

if __name__ == "__main__":
    test_graph()
    



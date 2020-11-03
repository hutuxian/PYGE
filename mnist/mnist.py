"""
-*- coding:utf-8 -*-
"""
import numpy as np
import ge


def check_ret(message, ret):
    """
    check return value
    """
    if ret != 0:
        raise Exception("{} failed ret = {}".format(message, ret))


def check_equal(message, val1, val2):
    """
    check return value
    """
    if val1 != val2:
        raise Exception("{} : {} != {}".format(message, val1, val2))


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
        # ge parameter initial
        self.ge_param = {
            "conv2D_32": {
                "fmt": ge.FORMAT_NHWC, "dt": ge.DT_FLOAT16,
                "w": None, "lr": 0.1, "strides": (1, 1, 1, 1), "pads": (2, 2, 2, 2),
                "w_shape": None, "w_shape_tensor": None, "in_shape": None,
                "var_name": ["conv2d_0_w", "conv2d_0_lr"],
                "var_value": None,
                "var_desc": None,
                "var_tensor": None,
                "x": None
            },
            "conv2D_64": {
                "fmt": ge.FORMAT_NHWC, "dt": ge.DT_FLOAT16,
                "w": None, "lr": 0.1, "strides": (1, 1, 1, 1), "pads": (2, 2, 2, 2),
                "w_shape": None, "w_shape_tensor": None, "in_shape": None,
                "var_name": ["conv2d_1_w", "conv2d_1_lr"],
                "var_value": None,
                "var_desc": None,
                "var_tensor": None,
                "x": None
            },
            "pool": {
                "fmt": ge.FORMAT_NHWC, "dt": ge.DT_FLOAT16,
                "in_shape_32": [1, 28, 28, 32], "in_shape_64": [1, 14, 14, 64],
                "strides": (1, 1, 1, 1), "padding": "VALID",
                "shape_tensor": None
            },
            "fc_1024": {
                "fmt": ge.FORMAT_NHWC, "dt": ge.DT_FLOAT16,
                "w": None, "w_shape": None, "b": None, "b_shape": None, "lr": 0.1,
                "var_name": ["fc_0_w", "fc_0_b"],
                "var_value": None,
                "var_desc": None,
                "var_tensor": None,
                "x": None
            },
            "fc_10": {
                "fmt": ge.FORMAT_NHWC, "dt": ge.DT_FLOAT16,
                "w": None, "w_shape": None, "b": None, "b_shape": None, "lr": 0.1,
                "var_name": ["fc_1_w", "fc_1_b"],
                "var_value": None,
                "var_desc": None,
                "var_tensor": None,
                "x": None
            },
            "softmax": {
                "fmt": ge.FORMAT_NHWC, "dt": ge.DT_FLOAT16
            },
            "cross_entropy": {
                "fmt": ge.FORMAT_NHWC, "dt": ge.DT_FLOAT16
            }
        }

    def __del__(self):
        ret = ge.ge_finalize()
        check_ret("ge_finalize", ret)
        print("Finalize ge success.")
        del self.session
        print("free session success.")

    def get_var_parameters(self):
        var_name, var_value, var_desc = [], [], []
        for i in self.ge_param.keys():
            if 'var_name' in self.ge_param[i]:
                for j in self.ge_param[i].keys():
                    if j == 'var_name':
                        var_name += self.ge_param[i][j]
                    elif j == 'var_value':
                        var_value += self.ge_param[i][j]
                    elif j == 'var_desc':
                        var_desc += self.ge_param[i][j]
        return var_name, var_value, var_desc

    def init_conv2d_param(self, filters, in_shape, w_shape):
        key = "conv2D_" + str(filters)
        if not self.ge_param[key]["w_shape"]:
            self.ge_param[key]["w_shape"] = w_shape
            self.ge_param[key]["w_shape_tensor"] = self.gen_tensor([len(self.ge_param[key]["w_shape"]), ],
                                                   self.ge_param[key]["w_shape"],
                                                   fmt=self.ge_param[key]["fmt"],
                                                   dt=ge.DT_INT32)

        if not self.ge_param[key]["w"]:
            self.ge_param[key]["w"] = self.gen_tensor_data(w_shape, -1, dt=self.ge_param[key]["dt"])

        if not self.ge_param[key]["lr"]:
            self.ge_param[key]["lr"] = 0.1

        if not self.ge_param[key]["strides"]:
            self.ge_param[key]["strides"] = (1, 1, 1, 1)

        if not self.ge_param[key]["pads"]:
            self.ge_param[key]["pads"] = (2, 2, 2, 2)

        self.ge_param[key]["in_shape"] = self.gen_tensor([4], in_shape, fmt=self.ge_param[key]["fmt"],
                                                         dt=ge.DT_INT32)

        if not self.ge_param[key]["var_desc"]:
            tensor_desc = ge.TensorDesc(ge.Shape(w_shape), self.ge_param[key]["fmt"], self.ge_param[key]["dt"])
            rate_desc = ge.TensorDesc(ge.Shape([1, ]), self.ge_param[key]["fmt"], self.ge_param[key]["dt"])
            self.ge_param[key]["var_desc"] = [tensor_desc, rate_desc]
            self.ge_param[key]["var_value"] = [self.ge_param[key]["w"], self.ge_param[key]["lr"]]

            self.ge_param[key]["var_tensor"] = [
                self.gen_tensor(w_shape, self.ge_param[key]["w"], fmt=self.ge_param[key]["fmt"],
                                dt=self.ge_param[key]["dt"]),
                self.gen_tensor([1, ], self.ge_param[key]["lr"], fmt=self.ge_param[key]["fmt"],
                                dt=self.ge_param[key]["dt"])
            ]

    def init_fc_param(self, n_out, w_shape, b_shape):
        key = "fc_" + str(n_out)
        if not self.ge_param[key]["w_shape"]:
            self.ge_param[key]["w_shape"] = w_shape

        if not self.ge_param[key]["b_shape"]:
            self.ge_param[key]["b_shape"] = b_shape

        if not self.ge_param[key]["w"]:
            self.ge_param[key]["w"] = self.gen_tensor_data(w_shape, -1, dt=self.ge_param[key]["dt"])

        if not self.ge_param[key]["b"]:
            self.ge_param[key]["b"] = self.gen_tensor_data(b_shape, 0.1, dt=self.ge_param[key]["dt"])

        if not self.ge_param[key]["var_desc"]:
            w_desc = ge.TensorDesc(ge.Shape(w_shape), self.ge_param[key]["fmt"], self.ge_param[key]["dt"])
            b_desc = ge.TensorDesc(ge.Shape(b_shape), self.ge_param[key]["fmt"], self.ge_param[key]["dt"])

            self.ge_param[key]["var_desc"] = [w_desc, b_desc]
            self.ge_param[key]["var_value"] = [self.ge_param[key]["w"], self.ge_param[key]["b"]]

            self.ge_param[key]["var_tensor"] = [
                self.gen_tensor(w_shape, self.ge_param[key]["w"], fmt=self.ge_param[key]["fmt"],
                                dt=self.ge_param[key]["dt"]),
                self.gen_tensor(b_shape, self.ge_param[key]["b"], fmt=self.ge_param[key]["fmt"],
                                dt=self.ge_param[key]["dt"])
            ]

    def init_pool_param(self):
        key = "pool"
        if not self.ge_param[key]["shape_tensor"]:
            self.ge_param[key]["shape_tensor"] = [
                self.gen_tensor([len(self.ge_param[key]["in_shape_32"]),],
                                self.ge_param[key]["in_shape_32"], fmt=self.ge_param[key]["fmt"],
                                dt=ge.DT_INT32),
                self.gen_tensor([len(self.ge_param[key]["in_shape_64"]),],
                                self.ge_param[key]["in_shape_64"], fmt=self.ge_param[key]["fmt"],
                                dt=ge.DT_INT32)
            ]

    def update_conv2d_params(self, filters, dw):
        key = "conv2D_" + str(filters)
        self.ge_param[key]["w"] -= self.ge_param[key]["lr"]*self.get_tensor_data(dw)
        self.ge_param[key]["var_tensor"][0] = self.gen_tensor(self.ge_param[key]["w_shape"],
                                                              self.ge_param[key]["w"],
                                                              fmt=self.ge_param[key]["fmt"],
                                                              dt=self.ge_param[key]["dt"])

    def update_fc_params(self, n_out, dw, db):
        key = "fc_" + str(n_out)
        self.ge_param[key]["w"] -= self.ge_param[key]["lr"]*self.get_tensor_data(dw)
        self.ge_param[key]["b"] -= self.ge_param[key]["lr"]*self.get_tensor_data(db)

        self.ge_param[key]["var_tensor"][0] = self.gen_tensor(self.ge_param[key]["w_shape"],
                                                              self.ge_param[key]["w"],
                                                              fmt=self.ge_param[key]["fmt"],
                                                              dt=self.ge_param[key]["dt"])
        self.ge_param[key]["var_tensor"][1] = self.gen_tensor(self.ge_param[key]["b_shape"],
                                                              self.ge_param[key]["b"],
                                                              fmt=self.ge_param[key]["fmt"],
                                                              dt=self.ge_param[key]["dt"])

    def update_fc_input(self, n_out, tensor):
        key = "fc_" + str(n_out)
        self.ge_param[key]["x"] = tensor

    def update_conv_input(self, filters, tensor):
        key = "conv2D_" + str(filters)
        self.ge_param[key]["x"] = tensor

    def update_input_params(self, forward_inputs, forward_outputs):
        self.update_fc_input(1024, forward_outputs[2])
        self.update_fc_input(10, forward_outputs[3])
        self.update_conv_input(64, forward_outputs[4])
        self.update_conv_input(32, forward_inputs[0])

    def update_net_params(self, backward_outputs):
        # fc_10_dw, fc_10_db, fc_1024_dw, fc_1024_db, conv2d_64_dw, conv2d_32_dw
        self.update_fc_params(1024, backward_outputs[2], backward_outputs[3])
        self.update_fc_params(10, backward_outputs[0], backward_outputs[1])
        self.update_conv2d_params(64, backward_outputs[4])
        self.update_conv2d_params(32, backward_outputs[5])

    def print_params(self):
        pass

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

    def print(self, inputs, outputs, data_type, print_data=False):
        for i in range(len(inputs)):
            data = self.get_tensor_data(inputs[i], data_type)
            print('size =', inputs[i].get_size(), inputs[i].get_tensor_desc().get_data_type(),
                  '\nx =', inputs[i].get_tensor_desc().get_shape().get_dims(), inputs[i].get_tensor_desc().get_format(),
                  '\ndata =', len(data), data.size, data.shape, "\n")
            if print_data:
                print(data)

        for i in range(len(outputs)):
            data = self.get_tensor_data(outputs[i], data_type)
            print('size =', outputs[i].get_size(), outputs[i].get_tensor_desc().get_data_type(),
                  '\nout =', len(data), data.size, data.shape,
                  '\nshape =', outputs[i].get_tensor_desc().get_shape().get_dims(),
                  outputs[i].get_tensor_desc().get_format(), "\n")
            if print_data:
                print(data)

    def reshape(self, tensor, shape):
        tensor_data = np.array(tensor.get_data(), dtype=np.uint8)
        tensor_desc = ge.TensorDesc(ge.Shape(shape),
                                    tensor.get_tensor_desc().get_format(),
                                    tensor.get_tensor_desc().get_data_type())
        tensor.set_tensor_desc(tensor_desc)
    
    def get_tensor_data(self, tensor, dtype=np.float16):
        data = np.array(tensor.get_data(), dtype=np.uint8)
        b_arr = data.tobytes()
        arr_2 = np.frombuffer(b_arr, dtype=dtype)
        return arr_2
    
    def get_tensor_from_bin(self, in_path, shape_list, fmt=ge.FORMAT_ND, dt=ge.DT_FLOAT16):
        size = 1
        for i in range(len(shape_list)):
            size *= shape_list[i]
        data_len = size * self.get_data_type_size(dt)
        np_in = np.fromfile(in_path, dtype=np.uint8)
        np_size = np_in.size * np_in.itemsize

        check_equal("get_tensor_from_bin:", np_size, data_len)
        input_tensor_desc = ge.TensorDesc(ge.Shape(shape_list), fmt, dt)
        input_tensor_desc.set_real_dim_cnt(len(shape_list))
        input_tensor = ge.Tensor(input_tensor_desc, np_in)
        return input_tensor

    def add_graph(self, graph_id, graph):
        ret = self.session.add_graph(graph_id, graph)
        check_ret("add_graph", ret)
        # print("Session add {} success.".format(graph_id))

    def run_graph(self, graph_id, in_tensor):
        out_tensor, ret = self.session.run_graph(graph_id, in_tensor)
        check_ret("run_graph", ret)
        # print("Session run {} success.".format(graph_id))
        ret = self.session.remove_graph(graph_id)
        check_ret("remove_graph", ret)
        # print("Session remove {} success.".format(graph_id))
        return out_tensor

    def check_rebuild(self, graph_id):
        if self.session.is_graph_need_rebuild(graph_id):
            self.session.remove_graph(graph_id)
            # print("Session check and remove {} success.".format(graph_id))

    def gen_tensor(self, tensor_shape, tensor_data, fmt=ge.FORMAT_NCHW, dt=ge.DT_FLOAT16):
        if dt == ge.DT_FLOAT:
            data_type = np.float32
        elif dt == ge.DT_FLOAT16:
            data_type = np.float16
        elif dt == ge.DT_INT32:
            data_type = np.int32
        else:
            data_type = np.uint8
        
        np_data = np.array(tensor_data, dtype=data_type)
        np_data = np.frombuffer(np_data.tobytes(), dtype=np.uint8)
        input_tensor_desc = ge.TensorDesc(ge.Shape(tensor_shape), fmt, dt)
        tensor = ge.Tensor(input_tensor_desc, np_data)

        return tensor
    
    def gen_tensor_data(self, tensor_shape, value, dt=ge.DT_FLOAT16):
        size = 1
        for i in range(len(tensor_shape)):
            size *= tensor_shape[i]

        if dt == ge.DT_FLOAT:
            data_type = np.float32
        elif dt == ge.DT_FLOAT16:
            data_type = np.float16
        else:
            data_type = np.uint8
        np_data = np.zeros(size, dtype=data_type)
        for i in range(size):
            np_data[i] = value

        return np_data
    
    def update_op_format(self, op, fmt):
        tensor_desc_x = op.get_input_desc("x")
        tensor_desc_y = op.get_output_desc("y")
        tensor_desc_x.set_format(fmt)
        tensor_desc_y.set_format(fmt)
        op.update_input_desc("x", tensor_desc_x)
        op.update_output_desc("y", tensor_desc_y)

    def update_op_datatype(self, op, dt):
        tensor_desc_x = op.get_input_desc("x")
        tensor_desc_y = op.get_output_desc("y")
        tensor_desc_x.set_data_type(dt)
        tensor_desc_y.set_data_type(dt)
        op.update_input_desc("x", tensor_desc_x)
        op.update_output_desc("y", tensor_desc_y)

    def update_op_desc(self, op, key):
        self.update_op_format(op, self.ge_param[key]["fmt"])
        self.update_op_datatype(op, self.ge_param[key]["dt"])

    def init_graph_param(self, graph, var_desc, var_name, var_values):
        inputs = []
        for i in range(len(var_desc)):
            var_desc[i].set_real_dim_cnt(var_desc[i].get_shape().get_dim_num())
            tensor = self.gen_tensor(var_desc[i].get_shape().get_dims(), var_values[i], fmt=var_desc[i].get_format(),
                                     dt=var_desc[i].get_data_type())
            tmp_var_name = var_name[i] + "_const_" + str(i)
            var_constant = ge.OperatorFactory.create_operator(tmp_var_name, "Constant").set_attr_tensor("value", tensor)
            var_constant.update_output_desc("y", var_desc[i])

            var_init = ge.OperatorFactory.create_operator(var_name[i], "Variable")
            var_init.update_output_desc("y", var_desc[i])
            tmp_var_name = var_name[i] + "_assign_" + str(i)
            ge.OperatorFactory.create_operator(tmp_var_name, "Assign").set_input("ref", var_init).set_input(
                "value", var_constant)
            inputs.append(var_init)
        graph.set_inputs(inputs)

    def ge_reshape(self, x, shape, name):
        in_shape_tensor = self.gen_tensor([len(shape),], shape, fmt=ge.FORMAT_ND, dt=ge.DT_INT32)
        const_shape = ge.OperatorFactory.create_operator(name + '_const_shape',
                                                         "Constant").set_attr_tensor("value", in_shape_tensor)
        const_shape.update_output_desc("y", in_shape_tensor.get_tensor_desc())
        reshape = ge.OperatorFactory.create_operator(name + "_reshape",
                                                     "Reshape").set_input("x", x) \
            .set_input("shape", const_shape)
        return const_shape, reshape

    def layer_conv2D(self, graph, x, filters):
        key = "conv2D_" + str(filters)
        var_w = ge.OperatorFactory.create_operator(self.ge_param[key]["var_name"][0], "Variable")
        var_w.update_output_desc("y", self.ge_param[key]["var_desc"][0])
        graph.add_op(var_w)
        conv2d = ge.OperatorFactory.create_operator(key, "Conv2D") \
            .set_input("x", x) \
            .set_input("filter", var_w) \
            .set_attr_vec_int64("strides", self.ge_param[key]["strides"]) \
            .set_attr_vec_int64("pads", self.ge_param[key]["pads"])
        self.update_op_desc(conv2d, key)
        tensor_desc_w = conv2d.get_input_desc("filter")
        tensor_desc_w.set_format(self.ge_param[key]["fmt"])
        conv2d.update_input_desc("filter", tensor_desc_w)
        graph.add_op(conv2d)
        return conv2d

    def layer_pool(self, graph, x, ksize=(1, 15, 15, 1)):
        key = "pool"
        op_name = key + str(ksize[1]) + str(ksize[2])
        avg_pool = ge.OperatorFactory.create_operator(op_name, "AvgPool") \
            .set_input("x", x) \
            .set_attr_vec_int64("ksize", ksize) \
            .set_attr_vec_int64("strides", self.ge_param[key]["strides"]) \
            .set_attr_string("padding", self.ge_param[key]["padding"])
        self.update_op_desc(avg_pool, key)
        graph.add_op(avg_pool)
        return avg_pool

    def layer_fc(self, graph, x, num_out=10, transpose_x1=False, transpose_x2=False):
        key = "fc_" + str(num_out)
        fc_w = ge.OperatorFactory.create_operator(self.ge_param[key]["var_name"][0], "Variable")
        fc_b = ge.OperatorFactory.create_operator(self.ge_param[key]["var_name"][1], "Variable")
        fc_w.update_output_desc("y", self.ge_param[key]["var_desc"][0])
        fc_b.update_output_desc("y", self.ge_param[key]["var_desc"][1])
        graph.add_op(fc_w)
        graph.add_op(fc_b)
        fc = ge.OperatorFactory.create_operator(key + "matmul_fc" ,
                                                "MatMul").set_input("x1", x) \
            .set_input("x2", fc_w) \
            .set_input("bias", fc_b) \
            .set_attr_bool("transpose_x1", transpose_x1) \
            .set_attr_bool("transpose_x2", transpose_x2)

        self.update_op_desc(fc, key)
        graph.add_op(fc)
        return fc

    def layer_softmax(self, graph, x):
        key = 'softmax'
        softmax = ge.OperatorFactory.create_operator("softmax",
                                                     "SoftmaxV2").set_input("x", x)
        self.update_op_desc(softmax, key)
        graph.add_op(softmax)
        return softmax

    def layer_loss(self, graph, features, labels):
        key = "cross_entropy"
        cross = ge.OperatorFactory.create_operator("softmax_cross_entropy_with_logits",
                                                   "SoftmaxCrossEntropyWithLogits").set_input("features", features) \
            .set_input("labels", labels)
        self.update_op_desc(cross, key)
        graph.add_op(cross)
        return cross

    def mnist_forward(self, graph):
        # conv2D 0
        data_x = ge.OperatorFactory.create_operator("conv2d_0_x", "Data").set_attr_int64("index", 0)
        conv2d_0 = self.layer_conv2D(graph, data_x, 32)
        # pool 0
        avg_pool_0 = self.layer_pool(graph, conv2d_0, ksize=(1, 15, 15, 1))
        # conv2D 1
        conv2d_1 = self.layer_conv2D(graph, avg_pool_0, 64)
        # pool 1
        avg_pool_1 = self.layer_pool(graph, conv2d_1, ksize=(1, 8, 8, 1))
        # fc 1024
        input_shapes = [1, 7*7*64]
        _, reshape_pool = self.ge_reshape(avg_pool_1, input_shapes, "fc_1024_forward/reshape/shape")
        fc_1024 = self.layer_fc(graph, reshape_pool, num_out=1024, transpose_x1=0, transpose_x2=0)
        # fc 10
        fc_10 = self.layer_fc(graph, fc_1024, num_out=10, transpose_x1=0, transpose_x2=0)
        # softmax
        soft_m = self.layer_softmax(graph, fc_10)
        # cross entropy
        data_labels = ge.OperatorFactory.create_operator("labels", "Data").set_attr_int64("index", 1)
        cross = self.layer_loss(graph, soft_m, data_labels)
        graph.set_inputs([data_x, data_labels]).set_outputs([cross, reshape_pool, fc_1024, avg_pool_0])

    def ge_matmul(self, name, x1, x2, b=None, transpose_x1=False, transpose_x2=False):
        matmul = ge.OperatorFactory.create_operator(name + "_matmul" ,
                                                    "MatMul").set_input("x1", x1) \
            .set_input("x2", x2) \
            .set_attr_bool("transpose_x1", transpose_x1) \
            .set_attr_bool("transpose_x2", transpose_x2)
        if b:
            matmul.set_input("bias", b)
        return matmul

    def layer_softmax_grad(self, graph, softmax, grad_softmax):
        key = "softmax"
        softmax_grad = ge.OperatorFactory.create_operator("soft_max_grad", "SoftmaxGrad") \
            .set_input("softmax", softmax) \
            .set_input("grad_softmax", grad_softmax)
        self.update_op_desc(softmax_grad, key)
        graph.add_op(softmax_grad)
        return softmax_grad

    def layer_fc_grad(self, graph, name, x1, x2, b=None, num_out=10, transpose_x1=False, transpose_x2=False):
        key = "fc_" + str(num_out)
        fc_grad = self.ge_matmul(key+'_'+name, x1, x2, b, transpose_x1=transpose_x1, transpose_x2=transpose_x2)
        self.update_op_desc(fc_grad, key)
        graph.add_op(fc_grad)
        return fc_grad

    def layer_pool_grad(self, graph, name, orig_input_shape, input_grad, ksize):
        key = "pool"
        avg_pool_grad = ge.OperatorFactory.create_operator(name+"_avg_pool_grad", "AvgPoolGrad") \
            .set_input("orig_input_shape", orig_input_shape) \
            .set_input("input_grad", input_grad) \
            .set_attr_vec_int64("ksize", ksize) \
            .set_attr_vec_int64("strides", self.ge_param[key]["strides"]) \
            .set_attr_string("padding", self.ge_param[key]["padding"])
        self.update_op_desc(avg_pool_grad, key)
        graph.add_op(avg_pool_grad)
        return avg_pool_grad

    def layer_conv2D_filter_grad(self, graph, filters, x, out_backprop):
        key = "conv2D_" + str(filters)
        filter_size = ge.OperatorFactory.create_operator(key+"_grad_filter/filter_size", "Constant") \
            .set_attr_tensor("value", self.ge_param[key]["w_shape_tensor"])
        graph.add_op(filter_size)
        conv2d_grad_filter = ge.OperatorFactory.create_operator(key + "_filter_grad",
                                                                "Conv2DBackpropFilter") \
            .set_input("x", x) \
            .set_input("out_backprop", out_backprop) \
            .set_attr_vec_int64("strides", self.ge_param[key]["strides"]) \
            .set_attr_vec_int64("pads", self.ge_param[key]["pads"]) \
            .set_input("filter_size", filter_size)
            # .set_attr_vec_int64("filter_size", self.ge_param[key]["var_desc"][0].get_shape().get_dims())
        self.update_op_desc(conv2d_grad_filter, key)
        graph.add_op(conv2d_grad_filter)
        return conv2d_grad_filter

    def layer_conv2D_x_grad(self, graph, filters, input_size, filter_w, out_backprop):
        key = "conv2D_" + str(filters)
        conv2d_x_grad = ge.OperatorFactory.create_operator(key + "_x_grad",
                                                           "Conv2DBackpropInput") \
            .set_input("input_size", input_size) \
            .set_input("filter", filter_w) \
            .set_input("out_backprop", out_backprop) \
            .set_attr_vec_int64("strides", self.ge_param[key]["strides"]) \
            .set_attr_vec_int64("pads", self.ge_param[key]["pads"])
        self.update_op_desc(conv2d_x_grad, key)
        graph.add_op(conv2d_x_grad)
        return conv2d_x_grad

    def mnist_backprop(self, graph):
        key = "softmax"
        softmax = ge.OperatorFactory.create_operator("soft_max_grad/softmax", "Data").set_attr_int64("index", 0)
        grad_softmax = ge.OperatorFactory.create_operator("soft_max_grad/grad_softmax",
                                                          "Data").set_attr_int64("index", 1)
        softmax_grad = self.layer_softmax_grad(graph, softmax, grad_softmax)
        # fc 10
        key = "fc_10"
        fc_10_x1 = ge.OperatorFactory.create_operator("fc_10_dw/x1",
                                                      "Constant").set_attr_tensor("value", self.ge_param[key]["x"])
        graph.add_op(fc_10_x1)
        # [1024, 1] x [1, 10] -> [1024, 10]
        fc_10_dw = self.ge_matmul("fc_10_dw", fc_10_x1, softmax_grad, transpose_x1=1, transpose_x2=0)
        fc_10_db = softmax_grad

        fc_10_w = ge.OperatorFactory.create_operator("fc_10_grad/x1", "Constant") \
            .set_attr_tensor("value", self.ge_param[key]["var_tensor"][0])
        fc_10_w.update_output_desc("y", self.ge_param[key]["var_tensor"][0].get_tensor_desc())
        graph.add_op(fc_10_w)
        # [1024, 10] x [1, 10] -> [1024, 1]
        fc_10_grad = self.layer_fc_grad(graph, "fc_10_grad", fc_10_w, softmax_grad, num_out=10, transpose_x1=0, transpose_x2=1)

        # fc 1024
        key = "fc_1024"
        fc_1024_x1 = ge.OperatorFactory.create_operator("fc_1024_dw/x1",
                                                        "Constant").set_attr_tensor("value", self.ge_param[key]["x"])
        fc_1024_x1.update_output_desc("y", self.ge_param[key]["x"].get_tensor_desc())
        graph.add_op(fc_1024_x1)
        # [1, 3136] x [1024, 1] -> [3136, 1024]
        fc_1024_dw = self.ge_matmul("fc_1024_dw", fc_1024_x1, fc_10_grad, transpose_x1=1, transpose_x2=1)
        fc_1024_db = fc_10_grad
        fc_1024_w = ge.OperatorFactory.create_operator("fc_1024_grad/x1", "Constant") \
            .set_attr_tensor("value", self.ge_param[key]["var_tensor"][0])
        fc_1024_w.update_output_desc("y", self.ge_param[key]["var_tensor"][0].get_tensor_desc())
        graph.add_op(fc_1024_w)
        # [1024, 1] x [3136, 1024] -> [1, 3136]
        fc_1024_grad = self.layer_fc_grad(graph, "fc_1024_grad", fc_10_grad, fc_1024_w, num_out=1024, transpose_x1=1, transpose_x2=1)

        # pool 1
        key = "pool"
        orig_input_shape = ge.OperatorFactory.create_operator("avg_pool_1_grad/orig_input_shape", "Constant") \
            .set_attr_tensor("value", self.ge_param[key]["shape_tensor"][1])
        orig_input_shape.update_output_desc("y", self.ge_param[key]["shape_tensor"][1].get_tensor_desc())
        graph.add_op(orig_input_shape)

        # [1, 3136] -> [1, 7, 7, 64] -> [1, 14, 14, 64]
        input_shapes = [1, 7, 7, 64]
        _, fc_1024_grad_reshape = self.ge_reshape(fc_1024_grad, input_shapes, "avg_pool_1_grad")
        self.update_op_desc(fc_1024_grad_reshape, key)
        graph.add_op(fc_1024_grad_reshape)
        avg_pool_1_grad = self.layer_pool_grad(graph, "avg_pool_1_grad", orig_input_shape,
                                               fc_1024_grad_reshape, (1, 8, 8, 1))
        
        # conv2D 1
        key = "conv2D_64"
        conv2d_64_x = ge.OperatorFactory.create_operator("conv2d_64_grad_filter/x",
                                                         "Constant") \
            .set_attr_tensor("value", self.ge_param[key]["x"])
        conv2d_64_x.update_output_desc("y", self.ge_param[key]["x"].get_tensor_desc())
        graph.add_op(conv2d_64_x)

        # [1, 14, 14, 64] x [64, 5, 5, 32] -> [1, 14, 14, 32]
        conv2d_64_dw = self.layer_conv2D_filter_grad(graph, 64, conv2d_64_x, avg_pool_1_grad)
        # [1, 28, 28, 64] x [64, 5, 5, 32]
        input_size_1 = ge.OperatorFactory.create_operator("conv2d_64_grad_in/input_size", "Constant") \
            .set_attr_tensor("value", self.ge_param[key]["in_shape"])
        input_size_1.update_output_desc("y", self.ge_param[key]["in_shape"].get_tensor_desc())
        graph.add_op(input_size_1)

        conv_w_1 = ge.OperatorFactory.create_operator("conv2d_64_grad_in/filter", "Constant") \
            .set_attr_tensor("value", self.ge_param[key]["var_tensor"][0])
        conv_w_1.update_output_desc("y", self.ge_param[key]["var_tensor"][0].get_tensor_desc())
        graph.add_op(conv_w_1)
        conv2d_64_grad_in = self.layer_conv2D_x_grad(graph, 64, input_size_1, conv_w_1, avg_pool_1_grad)

        # pool 0
        key = "pool"
        orig_input_shape = ge.OperatorFactory.create_operator("avg_pool_0_grad/orig_input_shape", 
                                                              "Constant") \
            .set_attr_tensor("value", self.ge_param[key]["shape_tensor"][0])
        orig_input_shape.update_output_desc("y", self.ge_param[key]["shape_tensor"][0].get_tensor_desc())
        graph.add_op(orig_input_shape)
        # [1, 14, 14, 32] -> [1, 28, 28, 32]
        avg_pool_0_grad = self.layer_pool_grad(graph, "avg_pool_0_grad", orig_input_shape,
                                               conv2d_64_grad_in, (1, 15, 15, 1))

        # conv2D 0 : in = [1, 28, 28, 1], out = [32, 5, 5, 1]
        key = "conv2D_32"
        conv2d_32_x = ge.OperatorFactory.create_operator("conv2d_32_grad_filter/x", "Constant") \
            .set_attr_tensor("value", self.ge_param[key]["x"])
        conv2d_32_x.update_output_desc("y", self.ge_param[key]["x"].get_tensor_desc())
        graph.add_op(conv2d_32_x)
        conv2d_32_dw = self.layer_conv2D_filter_grad(graph, 32, conv2d_32_x, avg_pool_0_grad)
        graph.add_op(conv2d_32_dw)

        graph.set_inputs([softmax, grad_softmax])\
            .set_outputs([fc_10_dw, fc_10_db, fc_1024_dw, fc_1024_db, conv2d_64_dw, conv2d_32_dw, conv2d_32_x, conv2d_64_x])

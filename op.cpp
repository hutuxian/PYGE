/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <graph/operator_reg.h>
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "state_ops.h"
#include "save_ops.h"
#include "nn_calculation_ops.h"
#include "nn_training_ops.h"

using namespace ge;
using namespace op;
namespace py = pybind11;

PYBIND11_MODULE(op, m) {
    m.doc() = "pybind11 op plugin"; // optional module docstring

    py::class_<Constant, Operator>(m, "Constant")
        .def(py::init<const string &>())
        .def(py::init<>())
        // OUTPUT
        .def("name_out_y", &Constant::name_out_y)
        .def("get_output_desc_y", &Constant::get_output_desc_y)
        .def("update_output_desc_y", &Constant::update_output_desc_y)
        // ATTR
        .def("name_attr_value", &Constant::name_attr_value)
        .def("get_attr_value", &Constant::get_attr_value)
        .def("set_attr_value", (Constant &(Constant::*)(const Tensor &)) &Constant::set_attr_value)
        .def("set_attr_value", (Constant &(Constant::*)(const function<Tensor()> &)) &Constant::set_attr_value);

    py::class_<Variable, Operator>(m, "Variable")
        .def(py::init<>())
        .def(py::init<const string &>())
        // INPUT
        .def("name_in_x", &Variable::name_in_x)
        .def("set_input_x", (Variable &(Variable::*)(Operator &, const string &)) &Variable::set_input_x)
        .def("set_input_x", (Variable &(Variable::*)(Operator &, uint32_t)) &Variable::set_input_x)
        .def("set_input_x", (Variable &(Variable::*)(Operator &)) &Variable::set_input_x)
        .def("get_input_desc_x", &Variable::get_input_desc_x)
        // OUTPUT
        .def("name_out_y", &Variable::name_out_y)
        .def("get_output_desc_y", &Variable::get_output_desc_y)
        .def("update_output_desc_y", &Variable::update_output_desc_y)
        // ATTR
        .def("name_attr_index", &Variable::name_attr_index)
        .def("get_attr_index", &Variable::get_attr_index)
        .def("set_attr_index", (Variable &(Variable::*)(const int64_t &)) &Variable::set_attr_index)
        .def("set_attr_index", (Variable &(Variable::*)(const function<int64_t()> &)) &Variable::set_attr_index)
        .def("name_attr_value", &Variable::name_attr_value)
        .def("get_attr_value", &Variable::get_attr_value)
        .def("set_attr_value", (Variable &(Variable::*)(const Tensor &)) &Variable::set_attr_value)
        .def("set_attr_value", (Variable &(Variable::*)(const function<Tensor()> &)) &Variable::set_attr_value)
        .def("name_attr_container", &Variable::name_attr_container)
        .def("get_attr_container", &Variable::get_attr_container)
        .def("set_attr_container", (Variable &(Variable::*)(const string &)) &Variable::set_attr_container)
        .def("set_attr_container", (Variable &(Variable::*)(const function<string()> &)) &Variable::set_attr_container);

    py::class_<Assign, Operator>(m, "Assign")
        .def(py::init<>())
        .def(py::init<const string &>())
        // INPUT
        .def("name_in_ref", &Assign::name_in_ref)
        .def("set_input_ref", (Assign &(Assign::*)(Operator &, const string &)) &Assign::set_input_ref)
        .def("set_input_ref", (Assign &(Assign::*)(Operator &, uint32_t)) &Assign::set_input_ref)
        .def("set_input_ref", (Assign &(Assign::*)(Operator &)) &Assign::set_input_ref)
        .def("get_input_desc_ref", &Assign::get_input_desc_ref)
        .def("name_in_value", &Assign::name_in_value)
        .def("set_input_value", (Assign &(Assign::*)(Operator &, const string &)) &Assign::set_input_value)
        .def("set_input_value", (Assign &(Assign::*)(Operator &, uint32_t)) &Assign::set_input_value)
        .def("set_input_value", (Assign &(Assign::*)(Operator &)) &Assign::set_input_value)
        .def("get_input_desc_value", &Assign::get_input_desc_value)
        // OUTPUT
        .def("name_out_ref", &Assign::name_out_ref)
        .def("get_output_desc_ref", &Assign::get_output_desc_ref)
        .def("update_output_desc_ref", &Assign::update_output_desc_ref)
        // ATTR
        .def("name_attr_validate_shape", &Assign::name_attr_validate_shape)
        .def("get_attr_validate_shape", &Assign::get_attr_validate_shape)
        .def("set_attr_validate_shape", (Assign &(Assign::*)(const bool &)) &Assign::set_attr_validate_shape)
        .def("set_attr_validate_shape", (Assign &(Assign::*)(const function<bool()> &)) &Assign::set_attr_validate_shape)
        .def("name_attr_use_locking", &Assign::name_attr_use_locking)
        .def("set_attr_use_locking", (Assign &(Assign::*)(const bool &)) &Assign::set_attr_use_locking)
        .def("set_attr_use_locking", (Assign &(Assign::*)(const function<bool()> &)) &Assign::set_attr_use_locking);

    py::class_<Data, Operator>(m, "Data")
        .def(py::init<>())
        .def(py::init<const string &>())
        // INPUT
        .def("name_in_x", &Data::name_in_x)
        .def("set_input_x", (Data &(Data::*)(Operator &, const string &)) &Data::set_input_x)
        .def("set_input_x", (Data &(Data::*)(Operator &, uint32_t)) &Data::set_input_x)
        .def("set_input_x", (Data &(Data::*)(Operator &)) &Data::set_input_x)
        .def("get_input_desc_x", &Data::get_input_desc_x)
        // OUTPUT
        .def("name_out_y", &Data::name_out_y)
        .def("get_output_desc_y", &Data::get_output_desc_y)
        .def("update_output_desc_y", &Data::update_output_desc_y)
        // ATTR
        .def("name_attr_index", &Data::name_attr_index)
        .def("get_attr_index", &Data::get_attr_index)
        .def("set_attr_index", (Data &(Data::*)(const int64_t &)) &Data::set_attr_index)
        .def("set_attr_index", (Data &(Data::*)(const function<int64_t()> &)) &Data::set_attr_index);

    py::class_<Add, Operator>(m, "Add")
        .def(py::init<>())
        .def(py::init<const string &>())
        // INPUT
        .def("name_in_x1", &Add::name_in_x1)
        .def("set_input_x1", (Add &(Add::*)(Operator &, const string &)) &Add::set_input_x1)
        .def("set_input_x1", (Add &(Add::*)(Operator &, uint32_t)) &Add::set_input_x1)
        .def("set_input_x1", (Add &(Add::*)(Operator &)) &Add::set_input_x1)
        .def("get_input_desc_x1", &Add::get_input_desc_x1)
        .def("name_in_x2", &Add::name_in_x2)
        .def("set_input_x2", (Add &(Add::*)(Operator &, const string &)) &Add::set_input_x2)
        .def("set_input_x2", (Add &(Add::*)(Operator &, uint32_t)) &Add::set_input_x2)
        .def("set_input_x2", (Add &(Add::*)(Operator &)) &Add::set_input_x2)
        .def("get_input_desc_x2", &Add::get_input_desc_x2)
        // OUTPUT
        .def("name_out_y", &Add::name_out_y)
        .def("get_output_desc_y", &Add::get_output_desc_y)
        .def("update_output_desc_y", &Add::update_output_desc_y);

}


cmake_minimum_required(VERSION 3.5)

project(ge LANGUAGES CXX)

#全局变量
#请将"/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64"替换为用到的.so所在的实际路径
set(ACL_LIB "/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/")
add_subdirectory(pybind11)

#安全规范编译命令
add_compile_options("-fPIE")
add_compile_options("-std=c++11")
add_compile_options("-fPIC")
add_compile_options("-fstack-protector-all")
add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=0")

add_link_options("-Wl,-z,now")
add_link_options("-Wl,-z,relro")
add_link_options("-Wl,-z,noexecstack")
add_link_options("-pie")
add_link_options("-s")
set(CMAKE_SKIP_RPATH TRUE)

#头文件包含
#请将"/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/include"替换为ge和graph文件夹所在的实际路径
include_directories(/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/include/)
include_directories(/usr/local/Ascend/ascend-toolkit/latest/opp/op_proto/built-in/inc/)

aux_source_directory(${PROJECT_SOURCE_DIR} APP_SOURCES)
#动态链接库
link_directories(${ACL_LIB})

link_libraries("libge_common.so" "libge_runner.so" "libgraph.so")

pybind11_add_module(ge ge.cpp)
pybind11_add_module(op op.cpp)

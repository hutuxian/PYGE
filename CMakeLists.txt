cmake_minimum_required(VERSION 3.5)

project(ge LANGUAGERS CXX)

#全局变量
#请将/usr/local/Ascend/ascend-toolkit/20.10.0.B020/x86_64-linux_gcc7.3.0/fwkacllib/lib64替换为用到的.so所在的实际路径
set(ACL_LIB "/usr/local/Ascend/ascend-toolkit/20.10.0.B020/x86_64-linux_gcc7.3.0/fwkacllib/lib64/")
add_subdirectory(pybind11)

#头文件包含
#请将/usr/local/Ascend/ascend-toolkit/20.10.0.B020/x86_64-linux_gcc7.3.0/fwkacllib/include替换为ge和graph文件夹所在的实际路径
include_directories(/usr/local/Ascend/ascend-toolkit/20.10.0.B020/x86_64-linux_gcc7.3.0/fwkacllib/include/)

aux_source_directory(${PROJECT_SOURCE_DIR} APP_SOURCES)
#动态链接库
link_directories(${ACL_LIB})

link_libraries("libge_common.so" "libge_runner.so" "libgraph.so")

pybind11_add_module(ge ge.cpp)

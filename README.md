# pyge

#### 介绍
The Python API for Graph Engine.

#### 依赖
开发与测试环境：A+X centos7.6（非强依赖）
依赖：Python3.7.5
      gcc 7.3.0

#### 使用说明
```
# 创建ge文件夹
mkdir ge
cd ge

# pybind11安装配置(linux)
git clone https://github.com/pybind/pybind11.git
cd pybind11

mkdir build
cd build
cmake ..
# 编译并运行测试用例
make check -j 4
# 用例可以正常运行，则证明pybind11安装成功
# 回到ge目录
cd ../..

# 编写ge.cpp对GE中的函数进行封装

# 编写CMakeLists.txt  注意更改其中的GE依赖的头文件与.so路径

# 编写cmake_build.sh

./cmake_build.sh

# 编译成功，生成封装好的ge.cpython-37m-x86_64-linux-gnu.so

# 设置环境变量
# 以下命令中的/usr/local/Ascend/ascend-toolkit/20.10.0.B020/fwkacllib以实际路径为准！
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/20.10.0.B020/fwkacllib/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/gcc7.3.0/lib64:/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/20.10.0.B020/fwkacllib/python/site-packages/te:$PYTHONPATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/20.10.0.B020/fwkacllib/python/site-packages/topi:$PYTHONPATH
export PATH=/usr/local/Ascend/ascend-toolkit/20.10.0.B020/fwkacllib/ccec_compiler/bin:$PATH
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/20.10.0.B020/opp

python3.7.5
import ge
# 运行成功没有报错就证明封装成功
# 然后就可以用封装好的ge函数编写对应的python脚本了
```

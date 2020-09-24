# pyge

## 介绍
The Python API for Graph Engine.

## 依赖

* 开发与测试环境：A+X centos7.6（非强依赖）<br>
* 依赖：Python3.7.5<br>
          gcc 7.3.0<br>
> 依赖详细信息详见《CANN 软件安装指南（训练）》

## 安装run包

* 从社区中下载所需的run包：
    * https://support.huawei.com/enterprise/zh/ascend-computing/atlas-data-center-solution-pid-251167910/software
    * https://support.huawei.com/enterprise/zh/ascend-computing/a800-9010-pid-250702809/software
* 非商用版本的最新版本获取：
    * A800-9010-npu-driver_20.1.0.B010_ubuntu18.04-x86_64.run
    * A800-9010-npu-firmware_1.75.t15.200.b150.run
    * Ascend-Toolkit-20.10.0.B023-x86_64-linux_gcc7.3.0.run
* 用root用户登录环境，上传所需的run包（比如：上传到`/home/hw`），安装步骤详见《CANN 软件安装指南（训练）》，此处仅简要说明
```
cd /home/hw
chmod 750 *.run
# 安装driver包
./A800-9010-npu-driver_20.1.0.B010_ubuntu18.04-x86_64.run --full
reboot
# 安装开发者套件包
./Ascend-Toolkit-20.10.0.B023-x86_64-linux_gcc7.3.0.run --install
```

## 使用说明
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
# 以下命令中的/usr/local/Ascend/ascend-toolkit/latest/fwkacllib以实际路径为准！
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/gcc7.3.0/lib64:/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/python/site-packages:$PYTHONPATH
export PATH=/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:$PATH
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp

python3.7.5
import ge
# 运行成功没有报错就证明封装成功
# 然后就可以用封装好的ge函数编写对应的python脚本了
```

## 许可证

[Apache License 2.0](LICENSE)

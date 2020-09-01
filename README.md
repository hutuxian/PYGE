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
#回到ge目录
cd ../..

# 编写ge.cpp对GE中的函数进行封装

# 编写CMakeLists.txt  注意更改其中的GE依赖的头文件与.so路径

# 编写cmake_build.sh

./cmake_build.sh

# 编译成功，生成封装好的ge.cpython-37m-x86_64-linux-gnu.so

#设置环境变量
# 以下命令中的/usr/local/Ascend/ascend-toolkit/20.10.0.B020/fwkacllib以实际路径为准！
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/20.10.0.B020/fwkacllib/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/gcc7.3.0/lib64:/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/20.10.0.B020/fwkacllib/python/site-packages/te:$PYTHONPATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/20.10.0.B020/fwkacllib/python/site-packages/topi:$PYTHONPATH
export PATH=/usr/local/Ascend/ascend-toolkit/20.10.0.B020/fwkacllib/ccec_compiler/bin:$PATH

python3.7.5
import ge
# 运行成功没有报错就证明封装成功
# 然后就可以用封装好的ge函数编写对应的python脚本了
```

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 码云特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  码云官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解码云上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是码云最有价值开源项目，是码云综合评定出的优秀开源项目
5.  码云官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  码云封面人物是一档用来展示码云会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)

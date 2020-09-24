#!/bin/bash

path_cur=$(cd `dirname $0`; pwd)
echo ${path_cur}
path_build=$path_cur/build

mkdir -p $path_build
cd $path_build
cmake ..
make
cp -af *.so ../

rm -rf $path_build
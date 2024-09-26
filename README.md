# chotagrad
Trying to Build Pytorch Tensor Library from sctrach

## compile and run test
`bash
gcc -O3 --shared -o tensor_lib.so -fPIC ops.c
`

`bash
g+++ -std=c++14 -shared -fPIC ops.cpp -o tensor_lib_cpp.so
`

`
uv run python_wrapper.py
`

## compile and run test using pybind11
`bash
python setup.py build_ext --inplace
`

`bash
uv run python_wrapper_cpp.py
`
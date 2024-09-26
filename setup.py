from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "tensor_lib",
        ["ops_pybind11.cpp"],
        extra_compile_args=['-std=c++14']
    ),
]

setup(
    name="tensor_lib",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

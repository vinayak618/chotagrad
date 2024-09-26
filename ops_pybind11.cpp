#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <random>

namespace py = pybind11;

class Tensor {
public:
    std::vector<float> data;
    std::vector<int> shape;

    Tensor(const std::vector<int>& shape) : shape(shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        data.resize(size);
    }

    void fill(float val) {
        std::fill(data.begin(), data.end(), val);
    }

    void random() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (float& elem : data) {
            elem = dis(gen);
        }
    }

    Tensor add(const Tensor& other) const {
        if (shape != other.shape) {
            throw std::runtime_error("Tensor shapes must match for addition");
        }
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    Tensor mul(const Tensor& other) const {
        if (shape != other.shape) {
            throw std::runtime_error("Tensor shapes must match for multiplication");
        }
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    float sum() const {
        return std::accumulate(data.begin(), data.end(), 0.0f);
    }

    Tensor exp() const {
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = std::exp(data[i]);
        }
        return result;
    }

    Tensor log() const {
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = std::log(data[i]);
        }
        return result;
    }
};

PYBIND11_MODULE(tensor_lib, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int>&>())
        .def("fill", &Tensor::fill)
        .def("random", &Tensor::random)
        .def("add", &Tensor::add)
        .def("mul", &Tensor::mul)
        .def("sum", &Tensor::sum)
        .def("exp", &Tensor::exp)
        .def("log", &Tensor::log)
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("shape", &Tensor::shape);
}

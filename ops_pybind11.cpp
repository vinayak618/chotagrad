#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <random>
#include <memory>

namespace py = pybind11;

template<typename T>
class Tensor {
public:
    std::unique_ptr<T[]> data;
    std::unique_ptr<int[]> shape;
    int ndim;
    int size;

    Tensor(const std::vector<int>& shape_vec) : ndim(shape_vec.size()) {
        shape = std::make_unique<int[]>(ndim);
        std::copy(shape_vec.begin(), shape_vec.end(), shape.get());

        size = 1;
        for (int i = 0; i < ndim; ++i) {
            size *= shape[i];
        }
        data = std::make_unique<T[]>(size);
    }

    void fill(T val) {
        std::fill(data.get(), data.get() + size, val);
    }

    void random() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (int i = 0; i < size; ++i) {
            data[i] = static_cast<T>(dis(gen));
        }
    }

    std::shared_ptr<Tensor<T>> add(const Tensor<T>& other) const {
        if (size != other.size) {
            throw std::runtime_error("Tensor sizes do not match for addition");
        }

        auto result = std::make_shared<Tensor<T>>(std::vector<int>(shape.get(), shape.get() + ndim));
        for (int i = 0; i < size; ++i) {
            result->data[i] = data[i] + other.data[i];
        }
        return result;
    }

    std::shared_ptr<Tensor<T>> mul(const Tensor<T>& other) const {
        if (size != other.size) {
            throw std::runtime_error("Tensor sizes do not match for multiplication");
        }

        auto result = std::make_shared<Tensor<T>>(std::vector<int>(shape.get(), shape.get() + ndim));
        for (int i = 0; i < size; ++i) {
            result->data[i] = data[i] * other.data[i];
        }
        return result;
    }

    T sum() const {
        return std::accumulate(data.get(), data.get() + size, static_cast<T>(0));
    }

    std::shared_ptr<Tensor<T>> exp() const {
        auto result = std::make_shared<Tensor<T>>(std::vector<int>(shape.get(), shape.get() + ndim));
        for (int i = 0; i < size; ++i) {
            result->data[i] = std::exp(data[i]);
        }
        return result;
    }

    std::shared_ptr<Tensor<T>> log() const {
        auto result = std::make_shared<Tensor<T>>(std::vector<int>(shape.get(), shape.get() + ndim));
        for (int i = 0; i < size; ++i) {
            result->data[i] = std::log(data[i]);
        }
        return result;
    }
};

PYBIND11_MODULE(tensor_lib, m) {
    py::class_<Tensor<float>, std::shared_ptr<Tensor<float>>>(m, "Tensor")
        .def(py::init<const std::vector<int>&>())
        .def("fill", &Tensor<float>::fill)
        .def("random", &Tensor<float>::random)
        .def("add", &Tensor<float>::add)
        .def("mul", &Tensor<float>::mul)
        .def("sum", &Tensor<float>::sum)
        .def("exp", &Tensor<float>::exp)
        .def("log", &Tensor<float>::log)
        .def_property_readonly("data", [](const Tensor<float>& t) {
            return std::vector<float>(t.data.get(), t.data.get() + t.size);
        })
        .def_property_readonly("shape", [](const Tensor<float>& t) {
            return std::vector<int>(t.shape.get(), t.shape.get() + t.ndim);
        });
}

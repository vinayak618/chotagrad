#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <numeric>

template<typename T>
struct Tensor {
    std::unique_ptr<T[]> data;
    std::unique_ptr<int[]> shape;
    int ndim;
    int size;
};

template<typename T>
class TensorWrapper {
private:
    std::shared_ptr<Tensor<T>> tensor;

public:
    TensorWrapper(const std::vector<int>& shape) {
        tensor = std::make_shared<Tensor<T>>();
        tensor->ndim = static_cast<int>(shape.size());
        tensor->shape = std::make_unique<int[]>(tensor->ndim);
        std::copy(shape.begin(), shape.end(), tensor->shape.get());

        tensor->size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        tensor->data = std::make_unique<T[]>(tensor->size);
    }

    // Copy constructor
    TensorWrapper(const TensorWrapper& other) : tensor(other.tensor) {}

    // Move constructor
    TensorWrapper(TensorWrapper&& other) noexcept : tensor(std::move(other.tensor)) {}

    // Copy assignment operator
    TensorWrapper& operator=(const TensorWrapper& other) {
        if (this != &other) {
            tensor = other.tensor;
        }
        return *this;
    }

    // Move assignment operator
    TensorWrapper& operator=(TensorWrapper&& other) noexcept {
        if (this != &other) {
            tensor = std::move(other.tensor);
        }
        return *this;
    }

    // Destructor is not needed due to smart pointers

    TensorWrapper operator+(const TensorWrapper& other) const {
        if (tensor->size != other.tensor->size) {
            throw std::runtime_error("Tensor sizes do not match for addition");
        }

        TensorWrapper result(std::vector<int>(tensor->shape.get(), tensor->shape.get() + tensor->ndim));
        for (int i = 0; i < tensor->size; ++i) {
            result.tensor->data[i] = tensor->data[i] + other.tensor->data[i];
        }
        return result;
    }

    TensorWrapper operator*(const TensorWrapper& other) const {
        if (tensor->size != other.tensor->size) {
            throw std::runtime_error("Tensor sizes do not match for multiplication");
        }

        TensorWrapper result(std::vector<int>(tensor->shape.get(), tensor->shape.get() + tensor->ndim));
        for (int i = 0; i < tensor->size; ++i) {
            result.tensor->data[i] = tensor->data[i] * other.tensor->data[i];
        }
        return result;
    }

    void print() const {
        std::cout << "Debug: Entering print method" << std::endl;
        std::cout << "Debug: ndim = " << tensor->ndim << ", size = " << tensor->size << std::endl;
        
        std::cout << "Tensor shape: (";
        for (int i = 0; i < tensor->ndim; ++i) {
            std::cout << tensor->shape.get()[i];
            if (i < tensor->ndim - 1) std::cout << ", ";
        }
        std::cout << ")\n";

        std::cout << "Data: ";
        for (int i = 0; i < tensor->size; ++i) {
            std::cout << tensor->data.get()[i] << " ";
        }
        std::cout << std::endl;  // Add this to ensure all data is flushed to output
        std::cout << std::endl;
        
        std::cout << "Debug: Exiting print method" << std::endl;
    }

    // Getter for the underlying Tensor pointer (for C compatibility)
    Tensor<T>* get_tensor() {
        return tensor.get();
    }

    // Setter for tensor data
    void set_data(const std::vector<T>& data) {
        if (static_cast<int>(data.size()) != tensor->size) {
            throw std::runtime_error("Data size does not match tensor size");
        }
        std::copy(data.begin(), data.end(), tensor->data.get());
    }
};

// C-compatible functions
extern "C" {
    TensorWrapper<float>* create_tensor_float(int* shape, int ndim) {
        try {
            std::cout << "Debug: Creating tensor with ndim = " << ndim << std::endl;
            std::vector<int> shape_vec(shape, shape + ndim);
            return new TensorWrapper<float>(shape_vec);
        } catch (const std::exception& e) {
            std::cerr << "Error creating tensor: " << e.what() << std::endl;
            return nullptr;
        }
    }

    void free_tensor_float(TensorWrapper<float>* tensor) {
        std::cout << "Debug: Freeing tensor" << std::endl;
        delete tensor;
    }

    void tensor_add_float(TensorWrapper<float>* a, TensorWrapper<float>* b, TensorWrapper<float>* result) {
        std::cout << "Debug: Performing tensor addition" << std::endl;
        try {
            *result = *a + *b;
        } catch (const std::exception& e) {
            std::cerr << "Error in tensor addition: " << e.what() << std::endl;
        }
    }

    void tensor_multiply_float(TensorWrapper<float>* a, TensorWrapper<float>* b, TensorWrapper<float>* result) {
        std::cout << "Debug: Performing tensor multiplication" << std::endl;
        try {
            *result = *a * *b;
        } catch (const std::exception& e) {
            std::cerr << "Error in tensor multiplication: " << e.what() << std::endl;
        }
    }

    void print_tensor_float(TensorWrapper<float>* tensor) {
        std::cout << "Debug: Entering print_tensor_float function" << std::endl;
        std::cout.flush();
        if (tensor) {
            Tensor<float>* t = tensor->get_tensor();
            std::cout << "Debug: Got tensor pointer" << std::endl;
            std::cout << "Debug: ndim = " << t->ndim << ", size = " << t->size << std::endl;
            std::cout.flush();
            
            std::cout << "Tensor shape: (";
            for (int i = 0; i < t->ndim; ++i) {
                std::cout << t->shape[i];
                if (i < t->ndim - 1) std::cout << ", ";
            }
            std::cout << ")" << std::endl;
            std::cout.flush();

            std::cout << "Data: ";
            for (int i = 0; i < t->size; ++i) {
                std::cout << t->data[i] << " ";
            }
            std::cout << std::endl;
            std::cout.flush();
        } else {
            std::cout << "Error: Null tensor pointer" << std::endl;
        }
        std::cout << "Debug: Exiting print_tensor_float function" << std::endl;
        std::cout.flush();
    }

    void set_tensor_data_float(TensorWrapper<float>* tensor, float* data, int size) {
        std::cout << "Debug: Entering set_tensor_data_float function" << std::endl;
        std::cout.flush();
        if (tensor) {
            std::vector<float> vec_data(data, data + size);
            tensor->set_data(vec_data);
            std::cout << "Debug: Data set successfully" << std::endl;
        } else {
            std::cout << "Error: Null tensor pointer in set_tensor_data_float" << std::endl;
        }
        std::cout << "Debug: Exiting set_tensor_data_float function" << std::endl;
        std::cout.flush();
    }
}

// Example usage
int main() {
    std::vector<int> shape = {2, 2};
    TensorWrapper<float> a(shape);
    TensorWrapper<float> b(shape);

    a.set_data({1.0f, 2.0f, 3.0f, 4.0f});
    b.set_data({5.0f, 6.0f, 7.0f, 8.0f});

    std::cout << "Tensor a:" << std::endl;
    a.print();
    std::cout << "Tensor b:" << std::endl;
    b.print();

    TensorWrapper<float> c = a + b;
    std::cout << "a + b:" << std::endl;
    c.print();

    TensorWrapper<float> d = a * b;
    std::cout << "a * b:" << std::endl;
    d.print();

    return 0;
}

#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <numeric>

struct Tensor {
    std::unique_ptr<float[]> data;
    std::unique_ptr<int[]> shape;
    int ndim;
    int size;
};

class TensorWrapper {
private:
    std::shared_ptr<Tensor> tensor;

public:
    TensorWrapper(const std::vector<int>& shape) {
        tensor = std::make_shared<Tensor>();
        tensor->ndim = static_cast<int>(shape.size());
        tensor->shape = std::make_unique<int[]>(tensor->ndim);
        std::copy(shape.begin(), shape.end(), tensor->shape.get());

        tensor->size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        tensor->data = std::make_unique<float[]>(tensor->size);
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
        // std::cout << "Debug: ndim = " << tensor->ndim << ", size = " << tensor->size << std::endl;
        
        std::cout << "Tensor shape: (";
        for (int i = 0; i < tensor->ndim; ++i) {
            std::cout << tensor->shape[i];
            if (i < tensor->ndim - 1) std::cout << ", ";
        }
        std::cout << ")\n";

        std::cout << "Data: ";
        for (int i = 0; i < tensor->size; ++i) {
            std::cout << tensor->data[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Debug: Exiting print method" << std::endl;
    }

    // Getter for the underlying Tensor pointer (for C compatibility)
    Tensor* get_tensor() {
        return tensor.get();
    }

    // Setter for tensor data
    void set_data(const std::vector<float>& data) {
        if (static_cast<int>(data.size()) != tensor->size) {
            throw std::runtime_error("Data size does not match tensor size");
        }
        std::copy(data.begin(), data.end(), tensor->data.get());
    }
};

// C-compatible functions
extern "C" {
    TensorWrapper* create_tensor(int* shape, int ndim) {
        try {
            std::cout << "Debug: Creating tensor with ndim = " << ndim << std::endl;
            std::vector<int> shape_vec(shape, shape + ndim);
            return new TensorWrapper(shape_vec);
        } catch (const std::exception& e) {
            std::cerr << "Error creating tensor: " << e.what() << std::endl;
            return nullptr;
        }
    }

    void free_tensor(TensorWrapper* tensor) {
        // std::cout << "Debug: Freeing tensor" << std::endl;
        delete tensor;
    }

    void tensor_add(TensorWrapper* a, TensorWrapper* b, TensorWrapper* result) {
        // std::cout << "Debug: Performing tensor addition" << std::endl;
        try {
            *result = *a + *b;
        } catch (const std::exception& e) {
            std::cerr << "Error in tensor addition: " << e.what() << std::endl;
        }
    }

    void tensor_multiply(TensorWrapper* a, TensorWrapper* b, TensorWrapper* result) {
        // std::cout << "Debug: Performing tensor multiplication" << std::endl;
        try {
            *result = *a * *b;
        } catch (const std::exception& e) {
            std::cerr << "Error in tensor multiplication: " << e.what() << std::endl;
        }
    }

    void print_tensor(TensorWrapper* tensor) {
        // std::cout << "Debug: Calling print_tensor function" << std::endl;
        if (tensor) {
            tensor->print();
        } else {
            std::cout << "Error: Null tensor pointer" << std::endl;
        }
    }
}

// Example usage
int main() {
    std::vector<int> shape = {2, 2};
    TensorWrapper a(shape);
    TensorWrapper b(shape);

    a.set_data({1, 2, 3, 4});
    b.set_data({5, 6, 7, 8});

    std::cout << "Tensor a:" << std::endl;
    a.print();
    std::cout << "Tensor b:" << std::endl;
    b.print();

    TensorWrapper c = a + b;
    std::cout << "a + b:" << std::endl;
    c.print();

    TensorWrapper d = a * b;
    std::cout << "a * b:" << std::endl;
    d.print();

    return 0;
}

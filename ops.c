#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float* data;
    int* shape;
    int ndim;
    int size;
} Tensor;

Tensor* create_tensor(int* shape, int ndim) {
    Tensor* tensor = (Tensor*) malloc(sizeof(Tensor));
    if(tensor == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for tensor\n");
        return NULL;
    }
    tensor->ndim = ndim;

    // Allocate memory for the shape array
    tensor->shape = (int*) malloc(ndim * sizeof(int));  // Fixed: Allocate ndim * sizeof(int)
    if(tensor->shape == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for the shape tensor\n");
        free(tensor);
        return NULL;
    }

    // Copy the input shape to the tensor shape array
    memcpy(tensor->shape, shape, ndim * sizeof(int));

    // Calculate the total size of the tensor
    tensor->size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->size *= shape[i];
    }

    // Allocate memory for the tensor data
    tensor->data = (float*)malloc(tensor->size * sizeof(float));
    if (tensor->data == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for tensor data\n");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    return tensor;
}

// Function to free the memory allocated for a tensor
void free_tensor(Tensor* tensor) {
    if (tensor != NULL) {
        free(tensor->data);   // Free the data array
        free(tensor->shape);  // Free the shape array
        free(tensor);         // Free the Tensor struct itself
    }
}

void tensor_add(Tensor* a, Tensor* b, Tensor* result) {
    // Check if the tensor sizes match
    if (a->size != b->size || a->size != result->size) {
        fprintf(stderr, "Error: Tensor sizes do not match for addition\n");
        return;
    }
    
    // Perform element-wise addition
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
}

// Function to multiply two tensors element-wise
void tensor_multiply(Tensor* a, Tensor* b, Tensor* result) {
    // Check if the tensor sizes match
    if (a->size != b->size || a->size != result->size) {
        fprintf(stderr, "Error: Tensor sizes do not match for multiplication\n");
        return;
    }
    
    // Perform element-wise multiplication
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
}

// Function to print the contents of a tensor
void print_tensor(Tensor* tensor) {
    // Print the shape of the tensor
    printf("Tensor shape: (");
    for (int i = 0; i < tensor->ndim; i++) {
        printf("%d", tensor->shape[i]);
        if (i < tensor->ndim - 1) printf(", ");
    }
    printf(")\n");
    
    // Print the data of the tensor
    printf("Data: ");
    for (int i = 0; i < tensor->size; i++) {
        printf("%.2f ", tensor->data[i]);
    }
    printf("\n");
}
// Guard to prevent multiple inclusions
#ifndef NEURAL_H
#define NEURAL_H

#include <vector>
#include <cstdlib> 
#include <iostream>
#include <ctime> 
#include <cmath>
#include <random> 
class Neural {
public:
    // Constructor: initializes the neural network layers and learning rate
    Neural(int inputSize, int hiddenSize, int outputSize, double learningRate);

    // Trains the neural network with given input and target output data
    void train(const std::vector<double>& input, const std::vector<double>& target);

    // Predicts the output given input data
    std::vector<double> predict(const std::vector<double>& input);

    // Prints the current state of the neural network weights
    void printWeights();

    // Prints the current state of the neural network biases
    void printBiases();

    // Backpropagates the error and updates weights and biases
    std::vector<double> backprop(const std::vector<double>& error,  const std::vector<double>& target);

private:
    int inputSize;      // Number of neurons in input layer
    int hiddenSize;     // Number of neurons in hidden layer
    int outputSize;     // Number of neurons in output layer
    double learningRate; // Learning rate for weight updates

    std::vector<int> inputLayer;        // Values of neurons in the input layer
    std::vector<double> hiddenLayer;       // Values of neurons in the hidden layer
    std::vector<double> outputLayer;       // Values of neurons in the output layer
    std::vector<double> hiddenDelta;       // Hidden layer of Delta 

    std::vector<std::vector<double>> weightsInputHidden;  // Weights between input and hidden layer
    std::vector<std::vector<double>> weightsHiddenOutput; // Weights between hidden and output layer

    std::vector<double> biasHidden;        // Biases of neurons in the hidden layer
    std::vector<double> biasOutput;        // Biases of neurons in the output layer

    // Forward propagates the input through the network
    std::vector<double> forward(const std::vector<double>& input);

   void initializeWeightsAndBiases();

    // Performs a single forward propagation step from one layer to the next
    std::vector<double> forwardStep(const std::vector<double>& input, 
                                    const std::vector<std::vector<double>>& weights, 
                                    const std::vector<double>& biases);

    // Calculates the error in the output layer (difference between predicted and actual values)
    std::vector<double> calculateOutputError(const std::vector<double>& target, const std::vector<double>& actual);

    // Performs the backpropagation algorithm, adjusting weights and biases
    void performBackpropagation(const std::vector<double>& outputError, 
                                const std::vector<double>& lastInput, 
                                const std::vector<double>& lastHidden,
                                const std::vector<double>& lastOutput);

    // Updates weights and biases based on the gradients derived during backpropagation
    void updateWeightsAndBiases(const std::vector<std::vector<double>>& deltaWeightsIH, 
                                const std::vector<std::vector<double>>& deltaWeightsHO, 
                                const std::vector<double>& deltaBiasesH, 
                                const std::vector<double>& deltaBiasesO);
    double sigmoid(double x);

    double sigmoidDerivative(double x);
};

#endif // NEURAL_H

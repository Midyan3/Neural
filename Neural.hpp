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
    /**
     * @brief Constructor that initializes the neural network layers and learning rate. (Default: printTraining = false, lambdaL1 = 0.01, lambdaL2 = 0.001)
     * 
     * @param inputSize Number of neurons in the input layer.
     * @param hiddenSize Number of neurons in the hidden layer.
     * @param outputSize Number of neurons in the output layer.
     * @param learningRate Learning rate for weight updates.
     */
    Neural(int inputSize, int hiddenSize, int outputSize, double learningRate);

    /**
     * @brief Trains the neural network with given input and target output data.
     * 
     * @param input Vector containing input data.
     * 
     * @param target Vector containing target output data.
     */
    void train(const std::vector<double>& input, const std::vector<double>& target);

    /**
     * @brief Predicts the output for given input data using the trained neural network.
     * 
     * @param input Vector containing input data.
     * 
     * @return std::vector<double> Predicted output.
     */
    std::vector<double> predict(const std::vector<double>& input);

    /**
     * @brief Prints the current state of the neural network weights.
     * 
     * @return void
     */
    void printWeights();

    /**
     * @brief Prints the current state of the neural network biases.
     * 
     * @return void
     */
    void printBiases();

    /**
     * @brief Set the Print Training Stats object
     * 
     * @param printTraining
     *  
     * @return void
     */
    void setPrintTrainingStats(bool printTraining);

    // Forward propagates the input through the network
    std::vector<double> forward(const std::vector<double>& input);
    
    // Backpropagates the error and updates weights and biases
      /**
     * @brief Backpropagates the error through the network and adjusts weights and biases.
     * 
     * @param error Vector containing error for each output neuron.
     * @param target Vector containing the expected output.
     * @return std::vector<double> Adjusted errors after backpropagation.
     */
    std::vector<double> backprop(const std::vector<double>& error,  const std::vector<double>& target);
    
    // Updates the Learning Rate of the network
    /**
     * @brief Set the Learning Rate object
     * 
     * @param learningRate 
     */
    void setLearningRate(double learningRate);
    
    // Applies L1 and L2 regularization to the weights
    /**
     * @brief Applies L1 and L2 regularization to the weights. Parameters are gradientsHiddenOutput and gradientsInputHidden and are both vectors of vectors of doubles.
     * 
     * @param gradientsHiddenOutput 
     * @param gradientsInputHidden 
     * 
     * @return void
     */
    void applyRegularization(std::vector<std::vector<double>>& gradientsHiddenOutput, std::vector<std::vector<double>>& gradientsInputHidden);

    // Sets the regularization factors. (Default: lambdaL1 = 0.01, lambdaL2 = 0.001)
    /**
     * @brief Set the Regularization object. Parameters are lambdaL1 and lambdaL2 and are both doubles.
     * 
     * @param lambdaL1 
     * @param lambdaL2 
     */
    void setRegularization(double lambdaL1, double lambdaL2);

    // Returns the current learning rate
    /**
     * @brief Get the Learning Rate object
     * 
     * @return double 
     */
    double getLearningRate();

    // Returns the current regularization factors. Returns a pair of doubles (lambdaL1, lambdaL2)
    /***
     * @brief Get the Regularization object
     * 
     * @return std::pair<double, double>
     * 
    */
    std::pair<double, double> getRegularization();

    // Returns the bool value of printTraining
    /**
     * @brief Get the Print Training Stats object
     * 
     * @return true 
     * @return false 
     */
    bool getPrintTrainingStats();




private:
    int inputSize;      // Number of neurons in input layer
    int hiddenSize;     // Number of neurons in hidden layer
    int outputSize;     // Number of neurons in output layer
    double learningRate; // Learning rate for weight updates
    double lambdaL1;  // regularization factor for L1
    double lambdaL2;  // regularization factor for L2
    bool printTraining; // Whether to print training statistics

    std::vector<double> inputLayer;        // Values of neurons in the input layer
    std::vector<double> hiddenLayer;       // Values of neurons in the hidden layer
    std::vector<double> outputLayer;       // Values of neurons in the output layer
    std::vector<double> hiddenDelta;       // Hidden layer of Delta 

    std::vector<std::vector<double>> weightsInputHidden;  // Weights between input and hidden layer
    std::vector<std::vector<double>> weightsHiddenOutput; // Weights between hidden and output layer

    std::vector<double> biasHidden;        // Biases of neurons in the hidden layer
    std::vector<double> biasOutput;        // Biases of neurons in the output layer


    /**
     * @brief Initializes the weights and biases of the neural network using Xavier initialization.
     * 
     * @return void
     * 
     */
    void initializeWeightsAndBiases();

    
    /**
     * @brief Calculates the sigmoid function applied to a value.
     * 
     * @param x The input value.
     * @return double The output of the sigmoid function.
     */
    double sigmoid(double x);
    
    //Prints training statistics
    void printTrainingStats(const std::vector<double>& error, const std::vector<double>& target, const std::vector<double>& results, const std::vector<double>& input);
    
     /**
     * @brief Calculates the derivative of the sigmoid function applied to a value.
     * 
     * @param x The input value.
     * @return double The derivative of the sigmoid function.
     */
    double sigmoidDerivative(double x);

     /**
     * @brief Calculates the ReLU function applied to a value.
     * 
     * @param x The input value.
     * @return double The output of the ReLU function.
     */
    double ReLU(double x);
    
    /**
     * @brief Calculates the derivative of the ReLU function applied to a value.
     * 
     * @param x The input value.
     * @return double The derivative of the ReLU function.
     */
    double ReLUDerivative(double x);

};

#endif // NEURAL_H

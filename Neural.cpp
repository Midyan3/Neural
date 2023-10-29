#include "Neural.hpp"

Neural::Neural(int inputSize, int hiddenSize, int outputSize, double learningRate) {
    try {
        this->inputSize = inputSize;
        this->hiddenSize = hiddenSize;
        this->outputSize = outputSize;
        this->learningRate = learningRate;
        printTraining = false;  // Default to not printing training stats
        lambdaL1 = 0.01;
        lambdaL2 = 0.001;

        initializeWeightsAndBiases();
    } catch (const std::exception &e) {
        std::cerr << "Exception in Neural constructor: " << e.what() << '\n';
    }
}
void Neural::initializeWeightsAndBiases(){ 
    try
    {   
        std::random_device rd;
        std::mt19937 generator(rd());

        // Xavier initialization for weights. I used this because it's more efficient than random initialization I had at the start.
        std::normal_distribution<double> distribution(0.0, std::sqrt(2.0 / (inputSize + hiddenSize)));

        weightsInputHidden.resize(inputSize, std::vector<double>(hiddenSize));
        weightsHiddenOutput.resize(hiddenSize, std::vector<double>(outputSize));
        biasHidden.resize(hiddenSize, 0.1);  // initialized with a small positive number to prevent dead neurons
        biasOutput.resize(outputSize, 0.1);

        for (size_t i = 0; i < inputSize; ++i) {
            for (size_t j = 0; j < hiddenSize; ++j) {
                weightsInputHidden[i][j] = distribution(generator);
            }
        }

        for (size_t i = 0; i < hiddenSize; ++i) {
            for (size_t j = 0; j < outputSize; ++j) {
                weightsHiddenOutput[i][j] = distribution(generator);
            }
        }

        hiddenLayer.resize(hiddenSize, 0.0);
        outputLayer.resize(outputSize, 0.0);

    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    

}
double Neural::sigmoid(double x){
    try
    {
        return 1.0 / (1.0 + exp(-x)); // Sigmoid function calculation
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    
}

double Neural::sigmoidDerivative(double x){
    try
    {

        return x * (1.0- x); // Derivative of the sigmoid function

    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 0.0;
    }
    
    
}   

void Neural::setRegularization(double lambdaL1, double lambdaL2){
    this->lambdaL1 = lambdaL1;
    this->lambdaL2 = lambdaL2;
}
void Neural::train(const std::vector<double>& input, const std::vector<double>& target){
    try{ 
        // Checks if the input and target vectors are of the correct size
        if(input.size() != inputSize || target.size() != outputSize){
            throw std::invalid_argument("Input or target vector size is incorrect");
        }

        inputLayer = input;// Directly assign the input vector

        // Perform a forward pass through the network and get the output
        std::vector<double> results = forward(input); 

        // Calculate the error for each output neuron
        std::vector<double> error(target.size());
        for(size_t i = 0; i < target.size(); ++i){
            error[i] = target[i] - results[i];
        }

        if(printTraining) printTrainingStats(error, target, results, input);

        // Perform backpropagation using the calculated error and target values
        backprop(error, target);  // 'backprop' should adjust the weights and biases in the network

    } catch (const std::exception& e) {
        std::cout << "Error in training: " << e.what() << std::endl;
    }
}

void Neural::printTrainingStats(const std::vector<double>& error, const std::vector<double>& target, const std::vector<double>& results, const std::vector<double>& input){
    // No need to catch exceptions here unless you're doing something specific that could throw
    std::cout << "---- Training Iteration ----" << std::endl;

    // Print inputs
    std::cout << "Input: [";
    for (size_t inp = 0; inp < input.size(); ++inp) {
        std::cout << input[inp];
        if (inp < input.size() - 1) std::cout << ", ";
    }
    std::cout << "]";

    // Printing target values
    std::cout << " - Target: [";
    for (size_t t = 0; t < target.size(); ++t) {
        std::cout << target[t];
        if (t < target.size() - 1) std::cout << ", ";
    }
    std::cout << "]";

    // Printing predicted results
    std::cout << " - Predicted: [";
    for (size_t r = 0; r < results.size(); ++r) {
        std::cout << results[r];
        if (r < results.size() - 1) std::cout << ", ";
    }
    std::cout << "]";

    // Printing error
    std::cout << " - Error: [";
    for (size_t e = 0; e < error.size(); ++e) {
        std::cout << error[e];
        if (e < error.size() - 1) std::cout << ", ";
    }
    std::cout << "]";

    std::cout << std::endl << "---------------------------" << std::endl;
}

void Neural::setPrintTrainingStats(bool printTraining){
    this->printTraining = printTraining;
}

std::vector<double> Neural::backprop(const std::vector<double>& error, const std::vector<double>& target){
    try{
        // Calculate DeltaOutput based on the provided error and the derivative of the activation function.
        std::vector<double> DeltaOutput(outputSize);
        for (int i = 0; i < outputSize; ++i) {
            double sigmoid_derivative = outputLayer[i] * (1 - outputLayer[i]); // derivative of the sigmoid
            DeltaOutput[i] = error[i] * sigmoid_derivative; // Adjust error signal for sigmoid derivative
        }

        // Calculate the hidden layer errors based on DeltaOutput and weightsHiddenOutput.
        std::vector<double> hiddenErrors(hiddenSize);
        for (int i = 0; i < hiddenSize; ++i) {
            hiddenErrors[i] = 0.0;
            for (int j = 0; j < outputSize; ++j) {
                hiddenErrors[i] += DeltaOutput[j] * weightsHiddenOutput[i][j];
            }
            hiddenErrors[i] *= hiddenLayer[i] > 0 ? 1.0 : 0.0; // ReLU derivative
        }

        // Updating weights and biases before regularization 
        std::vector<std::vector<double>> gradientsHiddenOutput(hiddenSize, std::vector<double>(outputSize));
        std::vector<std::vector<double>> gradientsInputHidden(inputSize, std::vector<double>(hiddenSize));

        for (int i = 0; i < hiddenSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                gradientsHiddenOutput[i][j] = DeltaOutput[j] * hiddenLayer[i];
            }
        }

        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < hiddenSize; ++j) {
                gradientsInputHidden[i][j] = hiddenErrors[j] * inputLayer[i];
            }
        }

        // Apply regularization on gradients
        applyRegularization(gradientsHiddenOutput, gradientsInputHidden);

        // Now, update the weights with the regularized gradients.
        for (int i = 0; i < hiddenSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                weightsHiddenOutput[i][j] -= learningRate * gradientsHiddenOutput[i][j];
            }
        }

        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < hiddenSize; ++j) {
                weightsInputHidden[i][j] -= learningRate * gradientsInputHidden[i][j];
            }
        }

        // Update biases for the hidden layer and the output layer.
        for (int i = 0; i < hiddenSize; ++i) {
            biasHidden[i] -= learningRate * hiddenErrors[i];
        }

        for (int i = 0; i < outputSize; ++i) {
            biasOutput[i] -= learningRate * DeltaOutput[i];
        }

        return hiddenErrors; 
    } catch (const std::exception& e) {
        std::cout << "Error in backpropagation: " << e.what() << std::endl;
        return std::vector<double>(); // Return an empty vector on exception
    }
}

void Neural::applyRegularization(std::vector<std::vector<double>>& gradientsHiddenOutput, std::vector<std::vector<double>>& gradientsInputHidden){
    // Regularize the gradients for the hidden-output layer.
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            gradientsHiddenOutput[i][j] += lambdaL1 * (weightsHiddenOutput[i][j] > 0 ? 1 : -1); // L1 Regularization
            gradientsHiddenOutput[i][j] += lambdaL2 * weightsHiddenOutput[i][j]; // L2 Regularization
        }
    }

    // Regularize the gradients for the input-hidden layer.
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            gradientsInputHidden[i][j] += lambdaL1 * (weightsInputHidden[i][j] > 0 ? 1 : -1); // L1 Regularization
            gradientsInputHidden[i][j] += lambdaL2 * weightsInputHidden[i][j]; // L2 Regularization
        }
    }
}

std::vector<double> Neural::predict(const std::vector<double>& input){
    try{
        if( input.size() != inputSize){
            throw std::invalid_argument("Input vector size is incorrect");
        }
        return forward(input); // Returns the output of the forward pass
    }catch (const std::exception& e) {
        std::cout << "Error in prediction: " << e.what() << std::endl;
        return std::vector<double>();
    }
}
double Neural::ReLU(double x){
    return std::max(0.0, x);  // ReLU returns 0 if x is negative, otherwise it returns x
}

double Neural::ReLUDerivative(double x){
    return x > 0.0 ? 1.0 : 0.0;  // The derivative of ReLU is 1 if x is positive, otherwise it's 0
}

bool Neural::getPrintTrainingStats(){
    return printTraining; // Returns whether or not to print training stats
}

double Neural::getLearningRate(){
    return learningRate; // Returns the learning rate
}

std::pair<double, double> Neural::getRegularization(){
    return std::make_pair(lambdaL1, lambdaL2); // Returns the regularization factors in a pair
}

std::vector<double> Neural::forward(const std::vector<double>& input) {
    try {
        if (input.size() != inputSize) {
            throw std::invalid_argument("Input vector size is incorrect");
        }
        std::fill(hiddenLayer.begin(), hiddenLayer.end(), 0.0);
        std::fill(outputLayer.begin(), outputLayer.end(), 0.0);
        // Calculate the input to the hidden layer neurons and apply the ReLU activation function
       for (int j = 0; j < hiddenSize; j++) {
            double activation = biasHidden[j];  
            for (int i = 0; i < inputSize; i++) {
                activation += input[i] * weightsInputHidden[i][j];  // add weighted inputs
            }
            hiddenLayer[j] = ReLU(activation);  // apply the ReLU activation function
        }


        outputLayer.clear();  // Makes sure the outputLayer is empty before assigning new values
        for (int j = 0; j < outputSize; j++) {
            double activation = biasOutput[j];  // start with the bias
            for (int i = 0; i < hiddenSize; i++) {
                activation += hiddenLayer[i] * weightsHiddenOutput[i][j];  // add weighted inputs
            }
            outputLayer.push_back(sigmoid(activation));  // applies the sigmoid activation function
        }

        return outputLayer;
    }
    catch(const std::exception& e) {
        std::cerr << "Exception in forward: " << e.what() << '\n';
        return std::vector<double>();  // Return an empty vector on exception
    }
}

void Neural::setLearningRate(double learningRate){
    this->learningRate = learningRate; // Sets the learning rate
}

void Neural::printBiases(){
    for(int i = 0; i < biasHidden.size(); i++){
        std::cout<<"BiasOutut at level " << i << " = "<<biasOutput[i]<<std::endl; 
        std::cout<<"BiasHidden at level " << i << " = "<<biasHidden[i]<<std::endl;
    }
}

void Neural::printWeights() {
    std::cout << "---- Weights: Input to Hidden ----" << std::endl;

    // Safety check
    if (weightsInputHidden.empty() || weightsInputHidden[0].empty()) {
        std::cout << "No weights available." << std::endl;
        return;
    }

    for (size_t i = 0; i < weightsInputHidden.size(); ++i) {
        for (size_t j = 0; j < weightsInputHidden[i].size(); ++j) {
            std::cout << "    " << weightsInputHidden[i][j]; 
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

}


/*
Forget about this down here, it's just a bunch of comments and stuff I used to help me understand the code.

  Data Samples:
  - {1,4,2,5}
  - {1,2,4,5}
  - {1,2,8,4}

  Subset Example:
  - {1,2,4}
  - {1}

  Calculation Breakdown:
  input  Hidden
  1        
          3   4        Output
  3                   0
          4   3
  4       4   2
        Hidden bias
         .01   .01

         .01   .01 

  Result Calculation:
  - Step 1: .72 + .2 = .92 (initial result)
  - Step 2: 1 - .92 = .08 (error calculation)
  - Step 3: .43 - .3 = .13 (adjusted result)

*/

//Created by Midyan Elghazali , Github: @Midyan3. Please feel free to use this code and if you have any modifications, feel free to share. This is my first neural network and I am still learning so any feedback is appreciated.
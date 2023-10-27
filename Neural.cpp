#include "Neural.hpp"

Neural::Neural(int inputSize, int hiddenSize, int outputSize, double learningRate) {
    try {
        this->inputSize = inputSize;
        this->hiddenSize = hiddenSize;
        this->outputSize = outputSize;
        this->learningRate = learningRate;

        initializeWeightsAndBiases();
    } catch (const std::exception &e) {
        std::cerr << "Exception in Neural constructor: " << e.what() << '\n';
    }
}
void Neural::initializeWeightsAndBiases(){ 
    try
    {   
        // Initialize random seed
        srand(static_cast<unsigned int>(time(nullptr)));
    // Initialize weights between input and hidden layer
        for (int i = 0; i < inputSize; i++){
        std::vector<double> weights;
        for (int j = 0; j < hiddenSize; j++){
            weights.push_back((double)rand() / RAND_MAX);
        }
        weightsInputHidden.push_back(weights);
    }

    // Initialize weights between hidden and output layer
    for (int i = 0; i < hiddenSize; i++){
        std::vector<double> weights;
        for (int j = 0; j < outputSize; j++){
            weights.push_back((double)rand() / RAND_MAX);
        }
        weightsHiddenOutput.push_back(weights);
    }

    // Initialize biases for hidden and output layer
    for (int i = 0; i < hiddenSize; i++){
        biasHidden.push_back((double)rand() / RAND_MAX);
    }
    for (int i = 0; i < outputSize; i++){
        biasOutput.push_back((double)rand() / RAND_MAX);
    }
     hiddenLayer = std::vector<double>(hiddenSize, 0.0);
    outputLayer = std::vector<double>(outputSize, 0.0);
    } 
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    

}
double Neural::sigmoid(double x){
    try
    {
        return 1.0 / (1.0 + exp(-x));
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    
}

double Neural::sigmoidDerivative(double x){
    try
    {
        return x * (1.0 - x);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    
}   
void Neural::printWeights(){
    for(auto x: weightsInputHidden){
        for(auto y: x){
            std::cout<<y<<" ";
        }
        std::cout<<std::endl;
    }
}

void Neural::train(const std::vector<double>& input, const std::vector<double>& target){
    try{ 
        // Checks if the input and target vectors are of the correct size
        if(input.size() != inputSize || target.size() != outputSize){
            throw std::invalid_argument("Input or target vector size is incorrect");
        }

        inputLayer.clear();  // Makes sure the inputLayer is empty before assigning new values
        for (const auto& value : input) {
            inputLayer.push_back(value);
        }

        // Perform a forward pass through the network and get the output
        std::vector<double> results = forward(input); 

        // Calculate the error for each output neuron
        std::vector<double> error(target.size());
        for(size_t i = 0; i < target.size(); i++){
            error[i] = target[i] - results[i];
            std::cout<<"-------------------"<<std::endl;
            std::cout<<"Input: "<<input[0]<<std::endl;
            std::cout<<"Neural result: "<< 1 - results[i]<<std::endl;
            std::cout<<"Correct result: "<<target[i]<<std::endl;
            std::cout << "error: " <<  error[i] << std::endl;
            std::cout<<"-------------------"<<std::endl;
        }

        // Perform backpropagation using the calculated error and target values
        backprop(error, target);  // 'backprop' should adjust the weights and biases in the network

    } catch (const std::exception& e) {
        std::cout << "Error in training: " << e.what() << std::endl;
    }
}

std::vector<double> Neural::backprop(const std::vector<double>& error, const std::vector<double>& target){
    try{
        // Calculates DeltaOutput based on the provided error and the derivative of the activation function.
        std::vector<double> DeltaOutput(outputSize);
        for (int i = 0; i < outputSize; ++i) {
            DeltaOutput[i] = error[i] * sigmoidDerivative(outputLayer[i]);
        }
        std::vector<double> hiddenErrors(hiddenSize);
        // Calculate the hidden layer errors based on DeltaOutput and weightsHiddenOutput.
        for (int i = 0; i < hiddenSize; ++i) {
            hiddenErrors[i] = 0.0;
            for (int j = 0; j < outputSize; ++j) {
                hiddenErrors[i] += DeltaOutput[j] * weightsHiddenOutput[i][j];
            }
            // Multiply the total error at each hidden neuron by the derivative of the activation function.
            hiddenErrors[i] *= sigmoidDerivative(hiddenLayer[i]);
        }

        // Update the weights for the hidden-output layer (weightsHiddenOutput).
        
        for (int i = 0; i < hiddenSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                double gradient = DeltaOutput[j] * hiddenLayer[i];
                weightsHiddenOutput[i][j] -= learningRate * gradient;
            }
        }

        // Update the weights for the input-hidden layer (weightsInputHidden).
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < hiddenSize; ++j) {
                double gradient = hiddenErrors[j] * inputLayer[i];
                weightsInputHidden[i][j] -= learningRate * gradient;
            }
        }

        // Update biases for the hidden layer and the output layer based on DeltaOutput and hiddenErrors.
        for (int i = 0; i < hiddenSize; ++i) {
            biasHidden[i] -= learningRate * hiddenErrors[i];
        }

        for (int i = 0; i < outputSize; ++i) {
            biasOutput[i] -= learningRate * DeltaOutput[i];
        }

        return hiddenErrors; 
    } catch (const std::exception& e) {
        std::cout << "Error in backpropagation: " << e.what() << std::endl;
        return std::vector<double>();
    }
}

std::vector<double> Neural::predict(const std::vector<double>& input){
    try{
        if( input.size() != inputSize){
            throw std::invalid_argument("Input vector size is incorrect");
        }
        for(int i = 0; i < inputSize; i++){
            inputLayer[i] = input[i];
        }
    }catch (const std::exception& e) {
        std::cout << "Error in prediction: " << e.what() << std::endl;
        return std::vector<double>();
    }
}


std::vector<double> Neural::forward(const std::vector<double>& input) {
    try
    {
     if (input.size() != inputSize) {
        throw std::invalid_argument("Input vector size is incorrect");
        }
        std::fill(hiddenLayer.begin(), hiddenLayer.end(), 0.0);
        std::fill(outputLayer.begin(), outputLayer.end(), 0.0);

    // calculate the input to the hidden layer neurons and apply the activation function
        std::vector<double> hiddenLayer(hiddenSize, 0.0);  // Initialize with zeros

        for (int j = 0; j < hiddenSize; j++) {
            double activation = biasHidden[j];  // start with the bias
            for (int i = 0; i < inputSize; i++) {
                activation += input[i] * weightsInputHidden[i][j];  // add weighted inputs
            }
            hiddenLayer[j] = sigmoid(activation);  // applies the activation function
         }
    outputLayer.clear();  // Makes sure the outputLayer is empty before assigning new values

    for(int j = 0; j < outputSize; j++){
        double activation = biasOutput[j];
        for(int i = 0; i < hiddenSize; i++){
            activation += hiddenLayer[i] * weightsHiddenOutput[i][j];
        }
        outputLayer.push_back(sigmoid(activation));
    }

    return outputLayer;
    }
    catch(const std::exception& e)
    {
          std::cerr << "Exception in forward: " << e.what() << '\n';
          return std::vector<double>();
    }

}

void Neural::printBiases(){
    for(int i = 0; i < biasHidden.size(); i++){
        std::cout<<"BiasOutut at level " << i << " = "<<biasOutput[i]<<std::endl; 
        std::cout<<"BiasHidden at level " << i << " = "<<biasHidden[i]<<std::endl;
    }
}


/*
{12}
{34r5[]}



*/

/*
{1,4,2,5}
{1,2,4,5}
{1,2,8,4}

{1,2,4}
{1}

1       4   
        3   4
3                   0
        4   3
4       4   2

       3    4

       4    4 

       .72 + .2 = .92 result 
         1 - .92 = .08 error
        .43 - .3 = .13 result

        "47"
        
*/




int main(){
    // Initialize your neural network with appropriate parameters
    Neural network(4 , 30, 1, 0.5);  // Assuming 1 input node, 5 hidden nodes, 2 output nodes, and a learning rate of 0.1

    // Seed random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);  // Assuming we want inputs in the range [0, 1]

    // Prepare a number of test cases   
        const int numTestCases = 1000;  // Or however many cases you'd like to test
        for(int i = 0; i < numTestCases; i++) {
       try
       {
           // Generate random input and target for diversity in test cases
        double random_input = dis(gen);
        std::vector<double> input = {36, 48, 91, 3};

        // You can generate targets based on any criteria fitting your test case needs
        // For demonstration, we're assuming a simple scenario where if input > 0.5 target is {1, 0} else {0, 1}
        std::vector<double> target = random_input > .5 ? std::vector<double>{1} : std::vector<double>{0};

        // Train the network with the generated input and target
        network.train(input, target);

        // Here, you might want to capture the network's output or error rates, or you may prefer to perform checks after all test cases
       }
       catch(const std::exception& e)
       {
        std::cerr << e.what() << '\n';
       }
        

    }

    // After training, you may want to evaluate your network's performance
    // This could involve passing a validation set through the network and checking the results against the expected outcomes

    // Output, logging, or further testing...

    return 0;
}
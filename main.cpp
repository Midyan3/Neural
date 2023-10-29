#include "Neural.cpp"
int main() {
    // Parameters
    int inputSize = 1;  // Our input is a single binary value
    int hiddenSize = 20; // Size of the hidden layer
    int outputSize = 1; // Our output is a single binary value
    double learningRate = 0.01; // Learning rate for training

    // Create the neural network
    Neural network(inputSize, hiddenSize, outputSize, learningRate);
    network.setPrintTrainingStats(true); // Print stats during training
    // Generate training data: simple binary input to binary output mapping
    std::vector<std::pair<std::vector<double>, std::vector<double>>> trainingData = {
        {{0}, {0}},
        {{1}, {1}}
    };
    int numEpochs = 2000; // Number of training epochs
    for(int epoch = 0; epoch < numEpochs; ++epoch) {
        for(const auto& sample : trainingData) {
            network.train(sample.first, sample.second); 
        }
    }
    std::cout << "Testing trained network:" << std::endl;
    for(const auto& test : trainingData) {
        std::vector<double> input = test.first;
        std::vector<double> predicted = network.predict(input); // Assuming 'forward' function does the forward pass and returns output
        double actual = test.second[0];

        // Round the predicted value to get a clear 0 or 1 classification
        std::cout << "Input: " << input[0] << ", Predicted: " << (predicted[0] > 0.5 ? 1 : 0) << ", Actual: " << actual << std::endl;
    }

    return 0;
}
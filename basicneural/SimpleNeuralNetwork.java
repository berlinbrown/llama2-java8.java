import java.util.Random;

public class SimpleNeuralNetwork {

    private static final int INPUT_NEURONS = 2;
    private static final int HIDDEN_NEURONS = 2;
    private static final int OUTPUT_NEURONS = 1;

    private double[][] inputToHiddenWeights;
    private double[][] hiddenToOutputWeights;
    private double[] hiddenBiases;
    private double[] outputBiases;

    private double[] hiddenLayer;
    private double[] outputLayer;

    private static final double LEARNING_RATE = 0.5;

    public SimpleNeuralNetwork() {
        Random rand = new Random();
        inputToHiddenWeights = new double[INPUT_NEURONS][HIDDEN_NEURONS];
        hiddenToOutputWeights = new double[HIDDEN_NEURONS][OUTPUT_NEURONS];
        hiddenBiases = new double[HIDDEN_NEURONS];
        outputBiases = new double[OUTPUT_NEURONS];
        hiddenLayer = new double[HIDDEN_NEURONS];
        outputLayer = new double[OUTPUT_NEURONS];

        // Initialize weights and biases
        for (int i = 0; i < INPUT_NEURONS; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                inputToHiddenWeights[i][j] = rand.nextDouble() * 2 - 1;
            }
        }
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < OUTPUT_NEURONS; j++) {
                hiddenToOutputWeights[i][j] = rand.nextDouble() * 2 - 1;
            }
        }
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            hiddenBiases[i] = rand.nextDouble() * 2 - 1;
        }
        for (int i = 0; i < OUTPUT_NEURONS; i++) {
            outputBiases[i] = rand.nextDouble() * 2 - 1;
        }
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }

    public double[] feedforward(double[] inputs) {
        // Calculate hidden layer activation
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            hiddenLayer[i] = 0;
            for (int j = 0; j < INPUT_NEURONS; j++) {
                hiddenLayer[i] += inputs[j] * inputToHiddenWeights[j][i];
            }
            hiddenLayer[i] += hiddenBiases[i];
            hiddenLayer[i] = sigmoid(hiddenLayer[i]);
        }

        // Calculate output layer activation
        for (int i = 0; i < OUTPUT_NEURONS; i++) {
            outputLayer[i] = 0;
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                outputLayer[i] += hiddenLayer[j] * hiddenToOutputWeights[j][i];
            }
            outputLayer[i] += outputBiases[i];
            outputLayer[i] = sigmoid(outputLayer[i]);
        }

        return outputLayer;
    }

    public void train(double[][] inputs, double[][] expectedOutputs, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int sample = 0; sample < inputs.length; sample++) {
                // Feedforward
                double[] input = inputs[sample];
                double[] targetOutput = expectedOutputs[sample];
                double[] actualOutput = feedforward(input);

                // Calculate output layer error
                double[] outputLayerError = new double[OUTPUT_NEURONS];
                double[] outputLayerDelta = new double[OUTPUT_NEURONS];
                for (int i = 0; i < OUTPUT_NEURONS; i++) {
                    outputLayerError[i] = targetOutput[i] - actualOutput[i];
                    outputLayerDelta[i] = outputLayerError[i] * sigmoidDerivative(actualOutput[i]);
                }

                // Calculate hidden layer error
                double[] hiddenLayerError = new double[HIDDEN_NEURONS];
                double[] hiddenLayerDelta = new double[HIDDEN_NEURONS];
                for (int i = 0; i < HIDDEN_NEURONS; i++) {
                    hiddenLayerError[i] = 0;
                    for (int j = 0; j < OUTPUT_NEURONS; j++) {
                        hiddenLayerError[i] += outputLayerDelta[j] * hiddenToOutputWeights[i][j];
                    }
                    hiddenLayerDelta[i] = hiddenLayerError[i] * sigmoidDerivative(hiddenLayer[i]);
                }

                // Update output layer weights and biases
                for (int i = 0; i < OUTPUT_NEURONS; i++) {
                    for (int j = 0; j < HIDDEN_NEURONS; j++) {
                        hiddenToOutputWeights[j][i] += LEARNING_RATE * outputLayerDelta[i] * hiddenLayer[j];
                    }
                    outputBiases[i] += LEARNING_RATE * outputLayerDelta[i];
                }

                // Update hidden layer weights and biases
                for (int i = 0; i < HIDDEN_NEURONS; i++) {
                    for (int j = 0; j < INPUT_NEURONS; j++) {
                        inputToHiddenWeights[j][i] += LEARNING_RATE * hiddenLayerDelta[i] * input[j];
                    }
                    hiddenBiases[i] += LEARNING_RATE * hiddenLayerDelta[i];
                }
            }
        }
    }

    public static void main(String[] args) {
        // XOR dataset
        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        double[][] outputs = {
            {0},
            {1},
            {1},
            {0}
        };

        SimpleNeuralNetwork nn = new SimpleNeuralNetwork();
        nn.train(inputs, outputs, 10000);

        // Test the neural network
        for (double[] input : inputs) {
            double[] output = nn.feedforward(input);
            System.out.printf("Input: [%f, %f] Output: %f\n", input[0], input[1], output[0]);
        }
    }
}

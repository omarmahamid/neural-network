# Neural Network from Scratch

This project implements a neural network from scratch in pure Java 21. The neural network is capable of training and testing using the MNIST dataset for digit recognition.


## Features

- **Neural Network Implementation**: The core of the project is the neural network implementation found in [`NeuralNetwork`](src/main/java/org/neural/network/neuralnetlib/net/NeuralNetwork.java).
- **Training and Testing**: The network can be trained and tested using the MNIST dataset. The training is done using stochastic gradient descent, implemented in [`StochasticGradientDescentTrainer`](src/main/java/org/neural/network/neuralnetlib/trainer/StochasticGradientDescentTrainer.java).
- **Cost Functions**: Different cost functions are available, such as the cross-entropy cost function in [`CrossEntropyCostFunction`](src/main/java/org/neural/network/neuralnetlib/options/cost/CrossEntropyCostFunction.java) and the quadratic cost function in [`QuadraticCostFunction`](src/main/java/org/neural/network/neuralnetlib/options/cost/QuadraticCostFunction.java).
- **Activation Functions**: The project includes various activation functions, such as the sigmoid function in [`SigmoidFunction`](src/main/java/org/neural/network/neuralnetlib/options/activation/SigmoidFunction.java).
- **Regularization**: Regularization techniques like L1 and L2 regularization are implemented in [`L1Regularization`](src/main/java/org/neural/network/neuralnetlib/options/regularization/L1Regularization.java) and [`L2Regularization`](src/main/java/org/neural/network/neuralnetlib/options/regularization/L2Regularization.java).
- **Data Handling**: Utilities for handling data, such as shuffling and subdividing datasets, are provided in [`DataUtils`](src/main/java/org/neural/network/neuralnetlib/net/DataUtils.java).
- **MNIST Data Loader**: The MNIST dataset loader is implemented in [`MNISTLoader`](src/main/java/org/neural/network/testermodule/MNISTLoader.java).
- **GUI**: A graphical user interface for drawing digits and interacting with the neural network is implemented in [`Frame`](src/main/java/org/neural/network/testermodule/Frame.java), [`ButtonPanel`](src/main/java/org/neural/network/testermodule/ButtonPanel.java), and [`DrawPanel`](src/main/java/org/neural/network/testermodule/DrawPanel.java).

## Getting Started

### Prerequisites

- Java 21
- Maven

### Building the Project

To build the project, run the following command in the root directory:

```sh
mvn clean install

mvn exec:java -Dexec.mainClass="org.neural.network.testermodule.Frame"
package org.neural.network.neuralnetlib.options.cost;


import org.neural.network.mathlib.algebra.Matrix;
import org.neural.network.mathlib.algebra.Vector;
import org.neural.network.neuralnetlib.net.NeuralNetwork;
import org.neural.network.neuralnetlib.options.activation.ActivationFunction;

/**
 * An abstract superclass for cost functions. Cost functions are used to analyze
 * how well your network is performing.
 *
 */
public abstract class CostFunction {

    /**
     * Calculates the total cost for some given test data.
     *
     * @param net network to test
     * @param dataIn input test data
     * @param dataOut output test data for evaluation
     * @return cost evaluated
     */
    public abstract double calculateTotal(NeuralNetwork net, Vector[] dataIn, Vector[] dataOut);

    /**
     * Calculates the error for one vector of training data.
     *
     * @param calcOut calculated output vector
     * @param dataOut output vector for evaluation
     * @param values values of network without activation applied
     * @param activationFunction activation function used in neural network
     * @return calculated error
     */
    public abstract Matrix calculateError(Matrix calcOut, Matrix dataOut, Matrix values, ActivationFunction activationFunction);
}
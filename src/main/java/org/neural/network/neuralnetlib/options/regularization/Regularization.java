package org.neural.network.neuralnetlib.options.regularization;


import org.neural.network.mathlib.algebra.Matrix;

/**
 * An abstract superclass for different regularization approaches.
 *
 */
public abstract class Regularization {

    /**
     * Applies the specified regularization on a matrix.
     *
     * @param weights current weights of neural network
     * @param learningRate the learning rate used in training the network
     * @param lambda the lambda that affects how intense the changes to the
     * weights are
     * @param n count of training data used for training
     * @return regularised weight array
     */
    public abstract Matrix calculate(Matrix weights, double learningRate, double lambda, int n);
}
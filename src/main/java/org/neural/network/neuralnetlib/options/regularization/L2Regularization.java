package org.neural.network.neuralnetlib.options.regularization;


import org.neural.network.mathlib.algebra.Matrix;

/**
 * Represents L2 regularization. This method multiplies every weight by
 * 1-((lambda*learningrate)/n). That way all weights are scaled towards zero.
 * This is useful for training because the weights do not saturate quickly. This
 * gives a boost in training speed.
 *
 */
public class L2Regularization extends Regularization {

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
    @Override
    public Matrix calculate(Matrix weights, double learningRate, double lambda, int n) {
        return weights.mulSca(1.0 - ((lambda * learningRate) / n));
    }

}
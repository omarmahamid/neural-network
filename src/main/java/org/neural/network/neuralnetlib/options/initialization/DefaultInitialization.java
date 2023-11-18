package org.neural.network.neuralnetlib.options.initialization;

import org.neural.network.mathlib.algebra.Matrix;
import org.neural.network.mathlib.algebra.Vector;

import java.util.Random;

/**
 * Initializes the weights and biases using gaussian normal distribution.
 * Performs no additional operations on the values.
 *
 */
public class DefaultInitialization extends WeightInitialization {

    /**
     * Initializes the bias vectors.
     *
     * @param sizes layer sizes of neural network
     * @return vector array with initialized values
     */
    @Override
    public Vector[] initBiases(int[] sizes) {
        Random rand = new Random();
        Vector[] biases = new Vector[sizes.length - 1];

        for (int i = 0; i < sizes.length - 1; i++) {
            double[] bias = new double[sizes[i + 1]];
            for (int j = 0; j < sizes[i + 1]; j++) {
                bias[j] = rand.nextGaussian();
            }
            biases[i] = new Vector(bias);
        }

        return biases;
    }

    /**
     * Initializes the weight matrices.
     *
     * @param sizes layer sizes of neural network
     * @return matrix array with initialized values
     */
    @Override
    public Matrix[] initWeights(int[] sizes) {
        Random rand = new Random();
        Matrix[] weights = new Matrix[sizes.length - 1];

        for (int i = 0; i < sizes.length - 1; i++) {
            double[][] weight = new double[sizes[i + 1]][sizes[i]];
            for (int j = 0; j < sizes[i + 1]; j++) {
                for (int k = 0; k < sizes[i]; k++) {
                    weight[j][k] = rand.nextGaussian();
                }
            }
            weights[i] = new Matrix(weight);
        }

        return weights;
    }

}
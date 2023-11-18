package org.neural.network.neuralnetlib.options.initialization;


import org.neural.network.mathlib.algebra.Matrix;
import org.neural.network.mathlib.algebra.Vector;


public abstract class WeightInitialization {

    /**
     * Initializes the bias vectors.
     *
     * @param sizes layer sizes of neural network
     * @return vector array with initialized values
     */
    public abstract Vector[] initBiases(int[] sizes);

    /**
     * Initializes the weight matrices.
     *
     * @param sizes layer sizes of neural network
     * @return matrix array with initialized values
     */
    public abstract Matrix[] initWeights(int[] sizes);
}
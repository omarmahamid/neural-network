package org.neural.network.neuralnetlib.options.activation;


import org.neural.network.mathlib.algebra.Matrix;
import org.neural.network.mathlib.algebra.Vector;

/**
 * Abstract superclass for activation functions that are being applied to
 * neurons.
 * */
public abstract class ActivationFunction {

    /**
     * Calculates activation for every entry of vector.
     *
     * @param v vector with data to calculate
     * @return calculated values as vector
     */
    public abstract Vector calculateVec(Vector v);

    /**
     * Calculates result for every entry of matrix.
     *
     * @param M matrix with data to calculate
     * @return calculated values as matrix
     */
    public abstract Matrix calculateMat(Matrix M);

    /**
     * Calculates derived activation for every entry of vector.
     *
     * @param v vector with data to calculate
     * @return calculated values as vector
     */
    public abstract Vector calculateDerivVec(Vector v);

    /**
     * Calculates derived activation for every entry of matrix.
     *
     * @param M matrix with data to calculate
     * @return calculated values as matrix
     */
    public abstract Matrix calculateDerivMat(Matrix M);
}
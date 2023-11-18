package org.neural.network.neuralnetlib.io;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.neural.network.mathlib.algebra.Matrix;
import org.neural.network.mathlib.algebra.Vector;
import org.neural.network.neuralnetlib.net.NeuralNetwork;
import org.neural.network.neuralnetlib.options.activation.ActivationFunction;
import org.neural.network.neuralnetlib.options.activation.SigmoidFunction;

/**
 * A helper class for saving and loading neural networks. Saving the network is
 * very important after you trained it for some time.
 *
 */
public class NetworkIO {

    private NetworkIO(){
        throw new IllegalStateException("Utility Class");
    }

    /**
     * Loads a neural network from the specified location.
     *
     * @param file filename of a neural network e.g. "network.dat"
     * @return the loaded neural network with all its weights and biases
     */
    public static NeuralNetwork loadNetwork(String file) throws Exception {
        Vector[] biases;
        Matrix[] weights;
        ActivationFunction activationFunction = null;
        int size = 0;
        try {
            List<String> lines = Files.readAllLines(Paths.get(file));
            if (lines.get(0).trim().equals("SIGMOID_ACTIVATION_FUNCTION")) {
                activationFunction = new SigmoidFunction();
            }
            size = Integer.parseInt(lines.get(1));
            biases = new Vector[size - 1];
            for (int i = 2; i < size + 1; i++) {
                String[] components = lines.get(i).trim().split(" ");
                double[] values = new double[components.length];
                for (int j = 0; j < components.length; j++) {
                    values[j] = Double.parseDouble(components[j]);
                }
                biases[i - 2] = new Vector(values);
            }
            weights = new Matrix[size - 1];
            int matrixIndex = 0;
            ArrayList<Double[]> rows = new ArrayList<>();
            for (int i = size + 1; i < lines.size(); i++) {
                if (lines.get(i).isEmpty()) {
                    double[][] matrix = new double[rows.size()][rows.get(0).length];
                    for (int j = 0; j < rows.size(); j++) {
                        for (int k = 0; k < rows.get(j).length; k++) {
                            matrix[j][k] = rows.get(j)[k];
                        }
                    }
                    rows.clear();
                    weights[matrixIndex++] = new Matrix(matrix);
                    i++;
                }
                if (i >= lines.size() || lines.get(i).isEmpty()) {
                    break;
                }
                String[] components = lines.get(i).trim().split(" ");
                Double[] values = new Double[components.length];
                for (int k = 0; k < components.length; k++) {
                    values[k] = Double.parseDouble(components[k]);
                }
                rows.add(values);

            }
        } catch (IOException ex) {
            throw new Exception("Exception while creating neural network");
        }
        return new NeuralNetwork(weights, biases, size, activationFunction);
    }

    /**
     * Saves the current state of a neural network. Can be imported again later
     * on.
     *
     * @param file the location and filename to save the network to
     * @param net the network that is to be saved
     */
    public static void saveNetwork(String file, NeuralNetwork net) {
        String functions = "";
        if (net.getActivationFunction() instanceof SigmoidFunction) {
            functions += "SIGMOID_ACTIVATION_FUNCTION\n";
        }
        StringBuilder text = new StringBuilder(functions + net.getSize() + "\n");
        for (Vector bias : net.getBiases()) {
            text.append(bias.toString()).append("\n");
        }
        for (Matrix weight : net.getWeights()) {
            text.append(weight.toString()).append("\n");
        }
        try {
            Files.write(Paths.get(file), text.toString().getBytes("UTF-8"));
        } catch (IOException ex) {
        }
    }
}
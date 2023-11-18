package org.neural.network.neuralnetlib;


import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.neural.network.neuralnetlib.io.NetworkIO;
import org.neural.network.neuralnetlib.net.NeuralNetwork;

class NetworkIOTest {



    @Test
    void unhappyPath() throws Exception {
        String fileName = "/Users/omarmahamid/Documents/GitHub/neural-network/src/test/resources/unhappy.dat";

        Assertions.assertThrows(Exception.class, () -> NetworkIO.loadNetwork(fileName));
    }


    @Test
    void testLoadNetwork() throws Exception {

        String fileName = "/Users/omarmahamid/Documents/GitHub/neural-network/src/test/resources/network.dat";
        NeuralNetwork network = NetworkIO.loadNetwork(fileName);

        Assertions.assertNotNull(network);

    }

}

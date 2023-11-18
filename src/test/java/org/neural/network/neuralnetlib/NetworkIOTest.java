package org.neural.network.neuralnetlib;


import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.neural.network.neuralnetlib.io.NetworkIO;
import org.neural.network.neuralnetlib.io.Resource;
import org.neural.network.neuralnetlib.net.NeuralNetwork;

class NetworkIOTest {



    @Test
    void unhappyPath() throws Exception {
        String fileName = new Resource("unhappy.dat").getAbsoluteFileName();

        Assertions.assertThrows(Exception.class, () -> NetworkIO.loadNetwork(fileName));
    }


    @Test
    void testLoadNetwork() throws Exception {

        String fileName = new Resource("network.dat").getAbsoluteFileName();
        NeuralNetwork network = NetworkIO.loadNetwork(fileName);

        Assertions.assertNotNull(network);

    }

}

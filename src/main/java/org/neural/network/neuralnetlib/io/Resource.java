package org.neural.network.neuralnetlib.io;

import java.net.URL;

public class Resource {
    private final URL resources;

    public Resource(String fileName){
        this.resources = Resource.class.getClassLoader().getResource(fileName);
    }

    public String getAbsoluteFileName(){
        return resources.getPath();
    }

}

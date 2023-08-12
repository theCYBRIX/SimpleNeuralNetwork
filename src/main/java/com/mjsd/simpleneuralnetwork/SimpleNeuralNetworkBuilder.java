package com.mjsd.simpleneuralnetwork;

import java.util.function.Function;

import com.google.gson.JsonSyntaxException;

import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.InputProvider;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.OutputHandler;

public class SimpleNeuralNetworkBuilder extends NeuralNetworkBuilder<SimpleNeuralNetwork>{
    final private static Function<NetworkLayout, SimpleNeuralNetwork> NETWORK_SUPPLIER = x -> new SimpleNeuralNetwork(x);

    public SimpleNeuralNetworkBuilder(){
        super(NETWORK_SUPPLIER);
    }

    public SimpleNeuralNetworkBuilder(SimpleNeuralNetwork initialState) throws NullPointerException{
        super(NETWORK_SUPPLIER, initialState);
    }

    public SimpleNeuralNetworkBuilder(NetworkLayout initialState) throws NullPointerException{
        super(NETWORK_SUPPLIER, initialState);
    }

    public SimpleNeuralNetworkBuilder(NetworkLayout initialState, InputProvider inputProvider, OutputHandler outputHandler) throws NullPointerException{
        super(NETWORK_SUPPLIER, initialState, inputProvider, outputHandler);
    }
    
    public static SimpleNeuralNetwork fromJson(String json) throws JsonSyntaxException{
        return NeuralNetworkBuilder.fromJson(json, SimpleNeuralNetwork.class);
    }
    
}

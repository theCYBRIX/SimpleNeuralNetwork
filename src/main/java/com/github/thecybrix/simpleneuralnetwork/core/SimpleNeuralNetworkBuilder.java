package com.github.thecybrix.simpleneuralnetwork.core;

import java.util.List;
import java.util.function.Function;

import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork.InputProvider;
import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork.OutputHandler;
import com.google.gson.JsonSyntaxException;

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

    public SimpleNeuralNetworkBuilder(SimpleNeuralNetworkBuilder initialState) throws NullPointerException{
        super(initialState);
    }

    public SimpleNeuralNetworkBuilder(NetworkLayout initialState, List<InputProvider> inputProvider, List<OutputHandler> outputHandler) throws NullPointerException{
        super(NETWORK_SUPPLIER, initialState, inputProvider, outputHandler);
    }

    public SimpleNeuralNetworkBuilder newBuilder(){
        return new SimpleNeuralNetworkBuilder(this);
    }
    
    public static SimpleNeuralNetwork fromJson(String json) throws JsonSyntaxException{
        return NeuralNetworkBuilder.fromJson(json, SimpleNeuralNetwork.class);
    }
    
}

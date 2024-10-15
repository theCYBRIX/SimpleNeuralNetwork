package com.github.thecybrix.simpleneuralnetwork.core;

import java.util.List;
import java.util.function.Function;

import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork.InputProvider;
import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork.OutputHandler;
import com.google.gson.JsonSyntaxException;

public class MutableNeuralNetworkBuilder extends NeuralNetworkBuilder<MutableNeuralNetwork>{
    final private static Function<NetworkLayout, MutableNeuralNetwork> NETWORK_SUPPLIER = x -> new MutableNeuralNetwork(x);

    public MutableNeuralNetworkBuilder(){
        super(NETWORK_SUPPLIER);
    }

    public MutableNeuralNetworkBuilder(MutableNeuralNetwork initialState) throws NullPointerException{
        super(NETWORK_SUPPLIER, initialState);
    }

    public MutableNeuralNetworkBuilder(NetworkLayout initialState) throws NullPointerException{
        super(NETWORK_SUPPLIER, initialState);
    }

    public MutableNeuralNetworkBuilder(MutableNeuralNetworkBuilder initialState) throws NullPointerException{
        super(initialState);
    }

    public MutableNeuralNetworkBuilder(NetworkLayout initialState, List<InputProvider> inputProvider, List<OutputHandler> outputHandler) throws NullPointerException{
        super(NETWORK_SUPPLIER, initialState, inputProvider, outputHandler);
    }

    public MutableNeuralNetworkBuilder newBuilder(){
        return new MutableNeuralNetworkBuilder(this);
    }
    
    public static MutableNeuralNetwork fromJson(String json) throws JsonSyntaxException{
        return NeuralNetworkBuilder.fromJson(json, MutableNeuralNetwork.class);
    }

}

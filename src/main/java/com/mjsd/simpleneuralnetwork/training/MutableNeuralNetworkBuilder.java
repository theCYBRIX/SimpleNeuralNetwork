package com.mjsd.simpleneuralnetwork.training;

import java.util.Collection;
import java.util.function.Function;

import com.google.gson.JsonSyntaxException;

import com.mjsd.simpleneuralnetwork.NetworkLayout;
import com.mjsd.simpleneuralnetwork.NeuralNetworkBuilder;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.InputProvider;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.OutputHandler;

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

    public MutableNeuralNetworkBuilder(NetworkLayout initialState, Collection<InputProvider> inputProvider, Collection<OutputHandler> outputHandler) throws NullPointerException{
        super(NETWORK_SUPPLIER, initialState, inputProvider, outputHandler);
    }
    
    public static MutableNeuralNetwork fromJson(String json) throws JsonSyntaxException{
        return NeuralNetworkBuilder.fromJson(json, MutableNeuralNetwork.class);
    }

}

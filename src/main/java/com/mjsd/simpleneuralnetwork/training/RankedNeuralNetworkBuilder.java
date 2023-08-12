package com.mjsd.simpleneuralnetwork.training;

import java.util.function.Function;

import com.google.gson.JsonSyntaxException;

import com.mjsd.simpleneuralnetwork.NetworkLayout;
import com.mjsd.simpleneuralnetwork.NeuralNetworkBuilder;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.InputProvider;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.OutputHandler;

public class RankedNeuralNetworkBuilder extends NeuralNetworkBuilder<RankedNeuralNetwork>{
    final private static Function<NetworkLayout, RankedNeuralNetwork> NETWORK_SUPPLIER = x -> new RankedNeuralNetwork(x);

    public RankedNeuralNetworkBuilder(){
        super(NETWORK_SUPPLIER);
    }

    public RankedNeuralNetworkBuilder(RankedNeuralNetwork initialState) throws NullPointerException{
        super(NETWORK_SUPPLIER, initialState);
    }

    public RankedNeuralNetworkBuilder(NetworkLayout initialState) throws NullPointerException{
        super(NETWORK_SUPPLIER, initialState);
    }

    public RankedNeuralNetworkBuilder(NetworkLayout initialState, InputProvider inputProvider, OutputHandler outputHandler) throws NullPointerException{
        super(NETWORK_SUPPLIER, initialState, inputProvider, outputHandler);
    }

    public static RankedNeuralNetwork fromJson(String json) throws JsonSyntaxException{
        return NeuralNetworkBuilder.fromJson(json, RankedNeuralNetwork.class);
    }

}

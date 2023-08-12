package com.mjsd.simpleneuralnetwork.gson;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import com.mjsd.simpleneuralnetwork.NetworkLayout;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork;
import com.mjsd.simpleneuralnetwork.NetworkLayout.NetworkLayer;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.ActivationFunction;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.InputNormalizer;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.InputProvider;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.OutputHandler;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.RankedNeuralNetwork;

public abstract class CustomGsonFactory {
    private static Gson customGson = null; 

    public static Gson getInstance(){
        if(customGson == null){
            GsonBuilder gsonBuilder = new GsonBuilder();
            gsonBuilder.registerTypeAdapter(ActivationFunction.class, new ActivationFunctionAdapter())
                       .registerTypeAdapter(InputNormalizer.class, new InputNormalizerAdapter())
                       .registerTypeAdapter(InputProvider.class, new InputProviderAdapter())
                       .registerTypeAdapter(OutputHandler.class, new OutputHandlerAdapter())
                       .registerTypeAdapter(NetworkLayer.class, new NetworkLayerAdapter())
                       .registerTypeAdapter(NetworkLayer[].class, new ArrayAdapter<>(NetworkLayer.class, NetworkLayer[]::new))
                       .registerTypeAdapter(NetworkLayout.class, new NetworkLayoutAdapter())
                       .registerTypeAdapter(SimpleNeuralNetwork.class, new SimpleNeuralNetworkAdapter())
                       .registerTypeAdapter(MutableNeuralNetwork.class, new MutableNeuralNetworkAdapter())
                       .registerTypeAdapter(RankedNeuralNetwork.class, new RankedNeuralNetworkAdapter());
            customGson = gsonBuilder.create();
        }

        return customGson;
    }
}

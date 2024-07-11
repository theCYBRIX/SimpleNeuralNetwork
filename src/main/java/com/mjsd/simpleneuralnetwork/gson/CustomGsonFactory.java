package com.mjsd.simpleneuralnetwork.gson;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import com.mjsd.simpleneuralnetwork.NetworkLayout;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork;
import com.mjsd.simpleneuralnetwork.NetworkLayout.NetworkLayer;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.ActivationFunction;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.InputNormalizer;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;

public final class CustomGsonFactory {

    private CustomGsonFactory(){}

    final public static Gson getInstance(){
        return CustomGsonHolder.CUSTOM_GSON;
    }

    protected static Gson createCustomGson(){
        GsonBuilder gsonBuilder = new GsonBuilder();
        gsonBuilder.registerTypeAdapter(ActivationFunction.class, new ActivationFunctionAdapter())
                    .registerTypeAdapter(InputNormalizer.class, new InputNormalizerAdapter())
                    .registerTypeAdapter(NetworkLayer.class, new NetworkLayerAdapter())
                    .registerTypeAdapter(NetworkLayer[].class, new ArrayAdapter<>(NetworkLayer.class, NetworkLayer[]::new))
                    .registerTypeAdapter(NetworkLayout.class, new NetworkLayoutAdapter())
                    .registerTypeAdapter(SimpleNeuralNetwork.class, new SimpleNeuralNetworkAdapter())
                    .registerTypeAdapter(MutableNeuralNetwork.class, new MutableNeuralNetworkAdapter());
        return gsonBuilder.create();
    }

    private static final class CustomGsonHolder{
        static final Gson CUSTOM_GSON = createCustomGson();
    }
}

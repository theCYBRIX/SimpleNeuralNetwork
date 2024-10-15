package com.github.thecybrix.simpleneuralnetwork.serialization.json;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayout;
import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayout.NetworkLayer;
import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork.ActivationFunction;
import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork.InputNormalizer;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

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

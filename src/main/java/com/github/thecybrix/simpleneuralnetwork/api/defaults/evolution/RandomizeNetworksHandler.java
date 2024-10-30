package com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution;

import com.github.thecybrix.simpleneuralnetwork.api.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.google.gson.JsonObject;

public class RandomizeNetworksHandler<E extends MutableNeuralNetwork> extends AbstractEvolutionRequestHandler<E> {
    final private static String DEFAULT_ENDPOINT = "randomize_networks";

    public RandomizeNetworksHandler(EvolutionContext<E> context) {
        super(context, DEFAULT_ENDPOINT);
    }

    public RandomizeNetworksHandler(EvolutionContext<E> context, String endpoint) {
        super(context, endpoint);
    }

    @Override
    public ResponsePacket handle(JsonObject request, EvolutionContext<E> context) throws Exception {
        context.randomizeNetworks();
        return ResponsePacket.ok();
    }
    
}

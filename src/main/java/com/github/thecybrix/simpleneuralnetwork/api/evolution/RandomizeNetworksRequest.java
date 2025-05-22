package com.github.thecybrix.simpleneuralnetwork.api.evolution;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.server.ContextualJsonRequestHandler;
import com.github.thecybrix.simpleneuralnetwork.server.ResponsePacket;
import com.google.gson.JsonObject;

public class RandomizeNetworksRequest<E extends MutableNeuralNetwork> extends ContextualJsonRequestHandler<EvolutionContext<E>> {
    final private static String DEFAULT_ENDPOINT = "randomizeNetworks";

    public RandomizeNetworksRequest(EvolutionContext<E> context) {
        this(context, DEFAULT_ENDPOINT);
    }

    public RandomizeNetworksRequest(EvolutionContext<E> context, String endpoint) {
        super(context, endpoint,
            //Required Properties
            NO_PROPERTIES,
            //Optional Properties
            NO_PROPERTIES,
            //Response Properties
            NO_PROPERTIES
        );
    }

    @Override
    public ResponsePacket handle(JsonObject request, EvolutionContext<E> context) throws Exception {
        context.randomizeNetworks();
        return ResponsePacket.ok();
    }
    
}

package com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution;

import java.util.Collections;

import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.api.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.google.gson.JsonObject;

public class GetBestNetworksHandler<E extends MutableNeuralNetwork> extends AbstractEvolutionRequestHandler<E> {
    final private static String DEFAULT_ENDPOINT = "get_best_networks";

    public GetBestNetworksHandler(EvolutionContext<E> context) {
        super(context, DEFAULT_ENDPOINT);
    }

    public GetBestNetworksHandler(EvolutionContext<E> context, String endpoint) {
        super(context, endpoint);
    }

    @Override
    public ResponsePacket handle(JsonObject request, EvolutionContext<E> context) throws Exception {
        RequestHandlerUtils.requireField(request, "numRequested");
        int numRequested = request.get("numRequested").getAsInt();

        if(context.getPreviousGeneration().isEmpty())
            throw new IllegalStateException("No network heirarchy available. This can happen if the server has not created a new generation since it was initialized or reconfigured.");
        
        
        return ResponsePacket.message(
            Collections.singletonMap(
                "networks",
                (numRequested > 0) ? context.getBestNetworks(numRequested) : context.getBestNetworks()
            )
        );
    }
    
}

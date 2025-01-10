package com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution;

import java.util.Collections;
import java.util.Map;

import com.github.thecybrix.simpleneuralnetwork.api.PropertyType;
import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.api.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.google.gson.JsonObject;

public class GetBestNetworksRequest<E extends MutableNeuralNetwork> extends AbstractEvolutionRequestHandler<E> {
    final private static String DEFAULT_ENDPOINT = "getBestNetworks";
    final private static String NUM_REQUESTED = "numRequested";
    final private static String NETWORKS = "networks";

    public GetBestNetworksRequest(EvolutionContext<E> context) {
        this(context, DEFAULT_ENDPOINT);
    }

    public GetBestNetworksRequest(EvolutionContext<E> context, String endpoint) {
        super(context, endpoint,
            //Required Properties
            Map.of(
                NUM_REQUESTED, PropertyType.INTEGER
            ),
            //Optional Properties
            NO_PROPERTIES
        );
    }

    @Override
    public ResponsePacket handle(JsonObject request, EvolutionContext<E> context) throws Exception {
        RequestHandlerUtils.requireField(request, NUM_REQUESTED);
        int numRequested = request.get(NUM_REQUESTED).getAsInt();

        if(context.getPreviousGeneration().isEmpty())
            throw new IllegalStateException("No network heirarchy available. This can happen if the server has not created a new generation since it was initialized or reconfigured.");
        
        
        return ResponsePacket.message(
            Collections.singletonMap(
                NETWORKS,
                (numRequested > 0) ? context.getBestNetworks(numRequested) : context.getBestNetworks()
            )
        );
    }
    
}

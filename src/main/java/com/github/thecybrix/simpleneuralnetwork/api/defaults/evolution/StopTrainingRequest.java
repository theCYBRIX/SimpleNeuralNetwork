package com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution;

import com.github.thecybrix.simpleneuralnetwork.api.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.google.gson.JsonObject;

public class StopTrainingRequest<E extends MutableNeuralNetwork> extends AbstractEvolutionRequestHandler<E> {
    final private static String DEFAULT_ENDPOINT = "stopTraining";

    public StopTrainingRequest(EvolutionContext<E> context) {
        this(context, DEFAULT_ENDPOINT);
    }

    public StopTrainingRequest(EvolutionContext<E> context, String endpoint) {
        super(context, endpoint,
            //Required Properties
            null,
            //Optional Properties
            null
        );
    }

    @Override
    public ResponsePacket handle(JsonObject request, EvolutionContext<E> context) throws Exception {
        context.stopTraining();
        return ResponsePacket.ok();
    }
    
}

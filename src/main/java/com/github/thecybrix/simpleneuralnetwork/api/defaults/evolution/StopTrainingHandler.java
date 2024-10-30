package com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution;

import com.github.thecybrix.simpleneuralnetwork.api.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.google.gson.JsonObject;

public class StopTrainingHandler<E extends MutableNeuralNetwork> extends AbstractEvolutionRequestHandler<E> {
    final private static String DEFAULT_ENDPOINT = "stop_training";

    public StopTrainingHandler(EvolutionContext<E> context) {
        super(context, DEFAULT_ENDPOINT);
    }

    public StopTrainingHandler(EvolutionContext<E> context, String endpoint) {
        super(context, endpoint);
    }

    @Override
    public ResponsePacket handle(JsonObject request, EvolutionContext<E> context) throws Exception {
        context.stopTraining();
        return ResponsePacket.ok();
    }
    
}

package com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution;

import java.util.Collections;
import java.util.HashMap;

import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.api.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

class NewGenerationHandler<E extends MutableNeuralNetwork> extends AbstractEvolutionRequestHandler<E>{
    final private static String DEFAULT_ENDPOINT = "create_new_generation";

    public NewGenerationHandler(EvolutionContext<E> context) {
        this(context, DEFAULT_ENDPOINT);
    }

    public NewGenerationHandler(EvolutionContext<E> context, String endpoint) {
        super(context, endpoint);
    }

    @Override
    public ResponsePacket handle(JsonObject request, EvolutionContext<E> context) throws Exception{
        HashMap<Integer, Double> networkScores = RequestHandlerUtils.GSON.fromJson(
            request.getAsJsonObject("networkScores"), 
            new TypeToken<HashMap<Integer, Double>>(){}.getType()
        );
        context.createNewGeneration(networkScores);
        return ResponsePacket.message(Collections.singletonMap("networkIDs", context.getCurrentGeneration().keySet()));
    }

}
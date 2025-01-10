package com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import com.github.thecybrix.simpleneuralnetwork.api.PropertyType;
import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.api.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

class CreateNewGenerationRequest<E extends MutableNeuralNetwork> extends AbstractEvolutionRequestHandler<E>{
    final private static String DEFAULT_ENDPOINT = "createNewGeneration";
    final private static String NETWORK_SCORES = "networkScores";
    final private static String NETWORK_IDS = "networkIds";

    public CreateNewGenerationRequest(EvolutionContext<E> context) {
        this(context, DEFAULT_ENDPOINT);
    }

    public CreateNewGenerationRequest(EvolutionContext<E> context, String endpoint) {
        super(context, endpoint,
            //Required Properties
            Map.of(
                NETWORK_SCORES, PropertyType.mapOf(PropertyType.INTEGER, PropertyType.DOUBLE)
            ),
            //Optional Properties
            NO_PROPERTIES
        );
    }

    @Override
    public ResponsePacket handle(JsonObject request, EvolutionContext<E> context) throws Exception{
        RequestHandlerUtils.requireField(request, NETWORK_SCORES);
        HashMap<Integer, Double> networkScores = RequestHandlerUtils.GSON.fromJson(
            request.getAsJsonObject(NETWORK_SCORES), 
            new TypeToken<HashMap<Integer, Double>>(){}.getType()
        );
        context.createNewGeneration(networkScores);
        return ResponsePacket.message(Collections.singletonMap(NETWORK_IDS, context.getCurrentGeneration().keySet()));
    }

}
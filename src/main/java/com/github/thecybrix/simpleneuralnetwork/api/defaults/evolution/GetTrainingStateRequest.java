package com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution;

import java.util.Map;

import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.api.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution.EvolutionContext.State;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.google.gson.JsonObject;

public class GetTrainingStateRequest<E extends MutableNeuralNetwork> extends AbstractEvolutionRequestHandler<E> {
    final private static String DEFAULT_ENDPOINT = "get_training_state";

    public GetTrainingStateRequest(EvolutionContext<E> context) {
        super(context, DEFAULT_ENDPOINT);
    }

    public GetTrainingStateRequest(EvolutionContext<E> context, String endpoint) {
        super(context, endpoint);
    }

    @Override
    public ResponsePacket handle(JsonObject request, EvolutionContext<E> context) throws Exception {

        State state = context.getTrainingState();
        
        long elapsedTime = context.getTrainingElapsedTime();
        double bestScore = context.getTrainingBestScore();
        int generation = context.getTrainingGeneration();
        double averageError = bestScore / context.getTrainingSampleCount();

        Map<Object, Object> trainingStatus = RequestHandlerUtils.map(
            "state", state.toString(),
            "elapsedTimeMS", elapsedTime,
            "averageError", averageError,
            "generation", generation
        );

        return ResponsePacket.message(trainingStatus);
    }
    
}

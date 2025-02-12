package com.github.thecybrix.simpleneuralnetwork.api.defaults.valuemapping;

import java.util.Map;

import com.github.thecybrix.simpleneuralnetwork.api.AbstractContextualRequestHandler;
import com.github.thecybrix.simpleneuralnetwork.api.PropertyType;
import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.api.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.api.defaults.valuemapping.ValueMappingContext.State;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.google.gson.JsonObject;

public class GetTrainingStateRequest<E extends MutableNeuralNetwork> extends AbstractContextualRequestHandler<ValueMappingContext<E>> {
    final private static String DEFAULT_ENDPOINT = "getTrainingState";
    final private static String STATE = "state";
    final private static String ELAPSED_TIME = "elapsedTimeMS";
    final private static String AVERAGE_ERROR = "averageError";
    final private static String GENERATION = "generation";

    public GetTrainingStateRequest(ValueMappingContext<E> context) {
        this(context, DEFAULT_ENDPOINT);
    }

    public GetTrainingStateRequest(ValueMappingContext<E> context, String endpoint) {
        super(context, endpoint,
            //Required Properties
            NO_PROPERTIES,
            //Optional Properties
            NO_PROPERTIES,
            //Response Properties
            Map.of(
                STATE, PropertyType.STRING,
                ELAPSED_TIME, PropertyType.LONG,
                AVERAGE_ERROR, PropertyType.DOUBLE,
                GENERATION, PropertyType.INTEGER
            )
        );
    }

    @Override
    public ResponsePacket handle(JsonObject request, ValueMappingContext<E> context) throws Exception {

        State state = context.getTrainingState();
        
        long elapsedTime = context.getTrainingElapsedTime();
        double bestScore = context.getTrainingBestScore();
        int generation = context.getTrainingGeneration();
        double averageError = bestScore / context.getTrainingSampleCount();

        Map<Object, Object> trainingStatus = RequestHandlerUtils.map(
            STATE, state.toString(),
            ELAPSED_TIME, elapsedTime,
            AVERAGE_ERROR, averageError,
            GENERATION, generation
        );

        return ResponsePacket.message(trainingStatus);
    }
    
}

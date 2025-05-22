package com.github.thecybrix.simpleneuralnetwork.api.valuemapping;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.server.ContextualJsonRequestHandler;
import com.github.thecybrix.simpleneuralnetwork.server.ResponsePacket;
import com.google.gson.JsonObject;

public class StopTrainingRequest<E extends MutableNeuralNetwork> extends ContextualJsonRequestHandler<ValueMappingContext<E>> {
    final private static String DEFAULT_ENDPOINT = "stopTraining";

    public StopTrainingRequest(ValueMappingContext<E> context) {
        this(context, DEFAULT_ENDPOINT);
    }

    public StopTrainingRequest(ValueMappingContext<E> context, String endpoint) {
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
    public ResponsePacket handle(JsonObject request, ValueMappingContext<E> context) throws Exception {
        context.stopTraining();
        return ResponsePacket.ok();
    }
    
}

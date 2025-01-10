package com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution;

import java.util.HashMap;
import java.util.Map;

import com.github.thecybrix.simpleneuralnetwork.api.PropertyType;
import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.api.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

public class ProcessInputsRequest<E extends MutableNeuralNetwork> extends AbstractEvolutionRequestHandler<E> {

    final private static String DEFAULT_ENDPOINT = "processInputs";
    final private static String NETWORK_INPUTS = "networkInputs";
    final private static String NETWORK_OUTPUTS = "networkOutputs";

    public ProcessInputsRequest(EvolutionContext<E> context) {
        this(context, DEFAULT_ENDPOINT);
    }

    public ProcessInputsRequest(EvolutionContext<E> context, String endpoint) {
        super(context, endpoint,
            //Required Properties
            Map.of(
                NETWORK_INPUTS, PropertyType.mapOf(PropertyType.INTEGER, PropertyType.arrayOf(PropertyType.DOUBLE))
            ),
            //Optional Properties
            NO_PROPERTIES
        );
    }

    @Override
    public ResponsePacket handle(JsonObject request, EvolutionContext<E> context) throws Exception {
        RequestHandlerUtils.requireField(request, NETWORK_INPUTS);
        HashMap<Integer, double[]> inputs = RequestHandlerUtils.GSON.fromJson(request.get(NETWORK_INPUTS), new TypeToken<HashMap<Integer, double[]>>(){}.getType());
        Map<Integer, double[]> outputs = context.processInputs(inputs);
        return ResponsePacket.message(NETWORK_OUTPUTS, outputs);
    }

}

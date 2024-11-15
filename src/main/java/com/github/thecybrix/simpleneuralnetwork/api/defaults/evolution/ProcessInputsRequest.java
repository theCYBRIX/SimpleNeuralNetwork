package com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution;

import java.util.HashMap;

import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.api.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

public class ProcessInputsRequest<E extends MutableNeuralNetwork> extends AbstractEvolutionRequestHandler<E> {

    final private static String DEFAULT_ENDPOINT = "process_inputs";

    public ProcessInputsRequest(EvolutionContext<E> context) {
        super(context, DEFAULT_ENDPOINT);
    }

    public ProcessInputsRequest(EvolutionContext<E> context, String endpoint) {
        super(context, endpoint);
    }

    @Override
    public ResponsePacket handle(JsonObject request, EvolutionContext<E> context) throws Exception {
        RequestHandlerUtils.requireField(request, "networkInputs");
        HashMap<Integer, double[]> inputs = RequestHandlerUtils.GSON.fromJson(request.get("networkInputs"), new TypeToken<HashMap<Integer, double[]>>(){}.getType());
        return ResponsePacket.message("networkOutputs", context.processInputs(inputs));
    }

}

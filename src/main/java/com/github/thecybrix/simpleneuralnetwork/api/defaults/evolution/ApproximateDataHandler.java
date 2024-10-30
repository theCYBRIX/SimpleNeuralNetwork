package com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution;

import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.api.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution.EvolutionContext.TrainingDataSet;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.google.gson.JsonObject;

public class ApproximateDataHandler<E extends MutableNeuralNetwork> extends AbstractEvolutionRequestHandler<E> {
    final private static String DEFAULT_ENDPOINT = "approximate_data_set";

    public ApproximateDataHandler(EvolutionContext<E> context) {
        super(context, DEFAULT_ENDPOINT);
    }

    public ApproximateDataHandler(EvolutionContext<E> context, String endpoint) {
        super(context, endpoint);
    }

    @Override
    public ResponsePacket handle(JsonObject request, EvolutionContext<E> context) throws Exception {
        TrainingDataSet dataSet = RequestHandlerUtils.GSON.fromJson(request, TrainingDataSet.class);
        context.approximateDataSet(dataSet);
        return ResponsePacket.ok();
    }
    
}

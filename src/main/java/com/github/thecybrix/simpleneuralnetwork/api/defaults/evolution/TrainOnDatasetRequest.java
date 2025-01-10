package com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution;

import java.util.Map;

import com.github.thecybrix.simpleneuralnetwork.api.PropertyType;
import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.api.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution.EvolutionContext.TrainingDataSet;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.google.gson.JsonObject;

public class TrainOnDatasetRequest<E extends MutableNeuralNetwork> extends AbstractEvolutionRequestHandler<E> {
    final private static String DEFAULT_ENDPOINT = "trainOnDataset";
    final private static String DATASET = "dataset";

    public TrainOnDatasetRequest(EvolutionContext<E> context) {
        this(context, DEFAULT_ENDPOINT);
    }

    public TrainOnDatasetRequest(EvolutionContext<E> context, String endpoint) {
        super(context, endpoint,
            //Required Properties
            Map.of(
                DATASET, PropertyType.of(PropertyType.OBJECT, PropertyType.arrayOf(PropertyType.DOUBLE ,2), PropertyType.arrayOf(PropertyType.DOUBLE ,2))
            ),
            //Optional Properties
            null
        );
    }

    @Override
    public ResponsePacket handle(JsonObject request, EvolutionContext<E> context) throws Exception {
        TrainingDataSet dataSet = RequestHandlerUtils.GSON.fromJson(request, TrainingDataSet.class);
        context.approximateDataSet(dataSet);
        return ResponsePacket.ok();
    }
    
}

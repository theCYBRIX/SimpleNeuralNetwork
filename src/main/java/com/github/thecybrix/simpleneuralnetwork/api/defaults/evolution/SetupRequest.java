package com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import com.github.thecybrix.simpleneuralnetwork.api.PropertyType;
import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils.ParentSelection;
import com.github.thecybrix.simpleneuralnetwork.api.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayout;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

public class SetupRequest<E extends MutableNeuralNetwork> extends AbstractEvolutionRequestHandler<E> {

    final private static String DEFAULT_ENDPOINT = "setup";
    final private static String NUM_NETWORKS = "numNetworks";
    final private static String PARENT_SELECTOR = "parentSelector";
    final private static String LAYOUT = "layout";
    final private static String CREATE_METADATA = "createMetadata";
    final private static String INITIAL_NETWORKS = "initialNetworks";
    final private static String NETWORK_IDS = "networkIds";
    

    public SetupRequest(EvolutionContext<E> context) {
        this(context, DEFAULT_ENDPOINT);
    }
    
    public SetupRequest(EvolutionContext<E> context, String endpoint) {
        super(context, endpoint,
            //Required Properties
            Map.of(
                NUM_NETWORKS, PropertyType.INTEGER,
                PARENT_SELECTOR, PropertyType.STRING
            ),
            //Optional Properties
            Map.of(
                LAYOUT, PropertyType.of("NetworkLayout"),
                CREATE_METADATA, PropertyType.BOOLEAN,
                INITIAL_NETWORKS, PropertyType.arrayOf("SimpleNeuralNetwork")
            )
        );
    }


    @Override
    public ResponsePacket handle(JsonObject request, EvolutionContext<E> context) throws Exception {
        int numNetworks;
        ParentSelection parentSelection;
        NetworkLayout layout = null;
        List<MutableNeuralNetwork> initialNetworks = null;
        boolean createMetadata = false;

        Objects.requireNonNull(request, "request is null");

        RequestHandlerUtils.requireFields(request, NUM_NETWORKS, PARENT_SELECTOR);

        numNetworks = request.get(NUM_NETWORKS).getAsInt();
        
        parentSelection = ParentSelection.valueOf(request.get(PARENT_SELECTOR).getAsString());

        if(request.has(LAYOUT))
            layout = RequestHandlerUtils.GSON.fromJson(request.getAsJsonObject(LAYOUT), NetworkLayout.class);
        
        if(request.has(CREATE_METADATA))
            createMetadata = request.get(CREATE_METADATA).getAsBoolean();
        
        if(request.has(INITIAL_NETWORKS))
            initialNetworks = RequestHandlerUtils.GSON.fromJson(request.get(INITIAL_NETWORKS), new TypeToken<ArrayList<MutableNeuralNetwork>>(){}.getType());

        context.setup(numNetworks, layout, parentSelection, initialNetworks, createMetadata);

        return ResponsePacket.message(Collections.singletonMap(NETWORK_IDS, context.getCurrentGeneration().keySet()));
    }
    
}

package com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils.ParentSelection;
import com.github.thecybrix.simpleneuralnetwork.api.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayout;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

public class SetupHanlder<E extends MutableNeuralNetwork> extends AbstractEvolutionRequestHandler<E> {

    final private static String DEFAULT_ENDPOINT = "setup";
    

    public SetupHanlder(EvolutionContext<E> context) {
        super(context, DEFAULT_ENDPOINT);
    }
    
    public SetupHanlder(EvolutionContext<E> context, String endpoint) {
        super(context, endpoint);
    }


    @Override
    public ResponsePacket handle(JsonObject request, EvolutionContext<E> context) throws Exception {
        int numNetworks;
        ParentSelection parentSelection;
        NetworkLayout layout;
        List<MutableNeuralNetwork> initialNetworks = null;

        Objects.requireNonNull(request, "request is null");

        RequestHandlerUtils.requireFields(request, "numNetworks", "parentSelector", "layout");

        numNetworks = request.get("numNetworks").getAsInt();
        
        parentSelection = ParentSelection.valueOf(request.get("parentSelector").getAsString());

        layout = RequestHandlerUtils.GSON.fromJson(request.getAsJsonObject("layout"), NetworkLayout.class);
        

        if(request.has("initialNetworks"))
            initialNetworks = RequestHandlerUtils.GSON.fromJson(request.get("initialNetworks"), new TypeToken<ArrayList<MutableNeuralNetwork>>(){}.getType());

        context.setup(numNetworks, layout, parentSelection, initialNetworks);

        return ResponsePacket.message(Collections.singletonMap("networkIDs", context.getCurrentGeneration().keySet()));
    }
    
}

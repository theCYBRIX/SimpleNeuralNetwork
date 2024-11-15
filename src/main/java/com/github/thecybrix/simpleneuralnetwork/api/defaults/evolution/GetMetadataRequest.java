package com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution;

import java.util.List;
import java.util.Map;

import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.api.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

public class GetMetadataRequest<E extends MutableNeuralNetwork> extends AbstractEvolutionRequestHandler<E>{
    final private static String DEFAULT_ENDPOINT = "get_metadata";
    
    public GetMetadataRequest(EvolutionContext<E> context) {
        super(context, DEFAULT_ENDPOINT);
    } 
    
    public GetMetadataRequest(EvolutionContext<E> context, String endpoint) {
        super(context, endpoint);
    } 

    @Override
    public ResponsePacket handle(JsonObject request, EvolutionContext<E> context) throws Exception {
        Map<Integer, Map<String, Object>> metadataPacket;

        if(request.has("networkIDs")){
            List<Integer> networkIDs = RequestHandlerUtils.GSON.fromJson(request.get("networkIDs"), new TypeToken<List<Integer>>(){}.getType());
            metadataPacket = context.getMetadata(networkIDs);
        } else {
            metadataPacket = context.getMetadata();
        }
        
        return ResponsePacket.message(metadataPacket);
    }
    
}

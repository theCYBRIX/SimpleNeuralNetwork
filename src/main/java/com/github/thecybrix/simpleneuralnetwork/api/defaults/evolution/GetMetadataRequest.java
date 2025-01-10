package com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution;

import java.util.List;
import java.util.Map;

import com.github.thecybrix.simpleneuralnetwork.api.PropertyType;
import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.api.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

public class GetMetadataRequest<E extends MutableNeuralNetwork> extends AbstractEvolutionRequestHandler<E>{
    final private static String DEFAULT_ENDPOINT = "getMetadata";
    final private static String NETWORK_IDS_KEY = "networkIds";
    
    public GetMetadataRequest(EvolutionContext<E> context) {
        this(context, DEFAULT_ENDPOINT);
    } 
    
    public GetMetadataRequest(EvolutionContext<E> context, String endpoint) {
        super(context, endpoint,
            //Required Properties
            NO_PROPERTIES,
            //Optional Properties
            Map.of(
                NETWORK_IDS_KEY, PropertyType.mapOf(PropertyType.INTEGER, PropertyType.mapOf(PropertyType.STRING, PropertyType.OBJECT))
            )
        );
    } 

    @Override
    public ResponsePacket handle(JsonObject request, EvolutionContext<E> context) throws Exception {
        Map<Integer, Map<String, Object>> metadataPacket;
        if(request.has(NETWORK_IDS_KEY)){
            List<Integer> networkIDs = RequestHandlerUtils.GSON.fromJson(request.get(NETWORK_IDS_KEY), new TypeToken<List<Integer>>(){}.getType());
            metadataPacket = context.getMetadata(networkIDs);
        } else {
            metadataPacket = context.getMetadata();
        }
        
        return ResponsePacket.message(metadataPacket);
    }
    
}

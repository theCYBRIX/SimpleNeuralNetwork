package com.github.thecybrix.simpleneuralnetwork.api.idmanager;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.server.ContextualJsonRequestHandler;
import com.github.thecybrix.simpleneuralnetwork.server.PropertyType;
import com.github.thecybrix.simpleneuralnetwork.server.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.server.ResponsePacket;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

public class GetMetadataRequest<E extends SimpleNeuralNetwork> extends ContextualJsonRequestHandler<NetworkIDManager<E>>{
    final private static String DEFAULT_ENDPOINT = "getMetadata";
    final private static String NETWORK_IDS_KEY = "networkIds";
    final private static String METADATA = "metadata";
    
    public GetMetadataRequest(NetworkIDManager<E> context) {
        this(context, DEFAULT_ENDPOINT);
    } 
    
    public GetMetadataRequest(NetworkIDManager<E> context, String endpoint) {
        super(context, endpoint,
            //Required Properties
            NO_PROPERTIES,
            //Optional Properties
            Map.of(
                NETWORK_IDS_KEY, PropertyType.arrayOf(PropertyType.INTEGER)
            ),
            //Response Properties
            Map.of(
                NETWORK_IDS_KEY, PropertyType.mapOf(PropertyType.INTEGER, PropertyType.mapOf(PropertyType.STRING, PropertyType.OBJECT))
            )
        );
    } 

    @Override
    public ResponsePacket handle(JsonObject request, NetworkIDManager<E> context) throws Exception {
        Map<Integer, Map<String, Object>> metadataPacket;
        if(request != null && request.has(NETWORK_IDS_KEY)){
            List<Integer> networkIDs = RequestHandlerUtils.GSON.fromJson(request.get(NETWORK_IDS_KEY), new TypeToken<List<Integer>>(){}.getType());
            metadataPacket = context.getMetadata(networkIDs);
        } else {
            metadataPacket = context.getMetadata();
        }
        
        return ResponsePacket.message(Collections.singletonMap(METADATA, metadataPacket));
    }
    
}

package com.github.thecybrix.simpleneuralnetwork.api.idmanager;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;

import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.server.ContextualJsonRequestHandler;
import com.github.thecybrix.simpleneuralnetwork.server.PropertyType;
import com.github.thecybrix.simpleneuralnetwork.server.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.server.ResponsePacket;
import com.google.gson.reflect.TypeToken;
import com.google.gson.JsonObject;


public class AddNetworkRequest<E extends SimpleNeuralNetwork> extends ContextualJsonRequestHandler<NetworkIDManager<E>> {
    final private static String DEFAULT_ENDPOINT = "addNetworks";
    final private static String NETWORK_IDS = "networkIds";
    final private static String NETWORKS = "networks";

    public AddNetworkRequest(NetworkIDManager<E> idManager){
        this(idManager, DEFAULT_ENDPOINT);
    }

    public AddNetworkRequest(NetworkIDManager<E> idManager, String endpoint){
        super(idManager, endpoint,
            //Required Properties
            Map.of(
                NETWORKS, PropertyType.of(SimpleNeuralNetwork[].class)
            ),
            //Optional Properties
            NO_PROPERTIES,
            //Response Properties
            Map.of(
                NETWORK_IDS, PropertyType.arrayOf(PropertyType.INTEGER)
            )
        );
    }

    //NOTE: Likely to break if actual child classes of SimpleNeuralNetwork are used
    @Override
    public ResponsePacket handle(JsonObject request, NetworkIDManager<E> context) throws Exception {
        ArrayList<E> networks = RequestHandlerUtils.GSON.fromJson(request.get(NETWORKS), new TypeToken<ArrayList<SimpleNeuralNetwork>>(){}.getType());
        int[] ids = context.addAll(networks);
        return ResponsePacket.message(Collections.singletonMap(NETWORK_IDS, ids));
    }
    
}

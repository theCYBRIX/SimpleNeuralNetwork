package com.github.thecybrix.simpleneuralnetwork.api.defaults.idmanager;

import java.util.ArrayList;
import java.util.Map;

import com.github.thecybrix.simpleneuralnetwork.api.AbstractContextualRequestHandler;
import com.github.thecybrix.simpleneuralnetwork.api.PropertyType;
import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.api.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

public class GetNetworkRequest<E extends SimpleNeuralNetwork> extends AbstractContextualRequestHandler<NetworkIDManager<E>> {
    final private static String DEFAULT_ENDPOINT = "getNetworks";
    final private static String NETWORK_IDS = "networkIds";
    final private static String NETWORKS = "networks";

    public GetNetworkRequest(NetworkIDManager<E> context) {
        this(context, DEFAULT_ENDPOINT);
    }

    public GetNetworkRequest(NetworkIDManager<E> context, String endpoint) {
        super(context, endpoint,
            //Required Properties
            NO_PROPERTIES,
            //Optional Properties
            Map.of(
                NETWORK_IDS, PropertyType.arrayOf(PropertyType.INTEGER)
            ),
            //Response Properties
            Map.of(
                NETWORKS, PropertyType.mapOf(PropertyType.INTEGER, PropertyType.of(SimpleNeuralNetwork.class))
            )
        );
    }

    @Override
    public ResponsePacket handle(JsonObject request, NetworkIDManager<E> context) throws Exception {
        RequestHandlerUtils.requireField(request, NETWORK_IDS);
        ArrayList<Integer> networkIds = RequestHandlerUtils.GSON.fromJson(request.get(NETWORK_IDS), new TypeToken<ArrayList<Integer>>(){}.getType());
        Map<Integer, E> outputs = context.getNetworks(networkIds);
        return ResponsePacket.message(NETWORKS, outputs);
    }
    
}

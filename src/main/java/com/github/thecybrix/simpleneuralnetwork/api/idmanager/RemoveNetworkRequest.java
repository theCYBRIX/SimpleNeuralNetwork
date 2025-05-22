package com.github.thecybrix.simpleneuralnetwork.api.idmanager;

import java.util.ArrayList;
import java.util.Map;

import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.server.ContextualJsonRequestHandler;
import com.github.thecybrix.simpleneuralnetwork.server.PropertyType;
import com.github.thecybrix.simpleneuralnetwork.server.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.server.ResponsePacket;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

public class RemoveNetworkRequest<E extends SimpleNeuralNetwork> extends ContextualJsonRequestHandler<NetworkIDManager<E>> {
    
    final private static String DEFAULT_ENDPOINT = "removeNetworks";
    final private static String NETWORK_IDS = "networkIds";

    public RemoveNetworkRequest(NetworkIDManager<E> idManager){
        this(idManager, DEFAULT_ENDPOINT);
    }

    public RemoveNetworkRequest(NetworkIDManager<E> idManager, String endpoint){
        super(idManager, endpoint,
            //Required Properties
            Map.of(
                NETWORK_IDS, PropertyType.arrayOf(PropertyType.INTEGER)
            ),
            //Optional Properties
            NO_PROPERTIES,
            //Response Properties
            NO_PROPERTIES
        );
    }

    @Override
    public ResponsePacket handle(JsonObject request, NetworkIDManager<E> context) throws Exception {
        ArrayList<Integer> networks = RequestHandlerUtils.GSON.fromJson(request.get(NETWORK_IDS), new TypeToken<ArrayList<Integer>>(){}.getType());
        context.removeAll(networks);
        return ResponsePacket.ok();
    }
}

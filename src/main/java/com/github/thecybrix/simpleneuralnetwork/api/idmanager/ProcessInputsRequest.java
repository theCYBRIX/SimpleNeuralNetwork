package com.github.thecybrix.simpleneuralnetwork.api.idmanager;

import java.util.HashMap;
import java.util.Map;

import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.server.AbstractJsonRequestHandler;
import com.github.thecybrix.simpleneuralnetwork.server.PropertyType;
import com.github.thecybrix.simpleneuralnetwork.server.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.server.ResponsePacket;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;

public class ProcessInputsRequest<E extends SimpleNeuralNetwork> extends AbstractJsonRequestHandler {

    final private static String DEFAULT_ENDPOINT = "processInputs";
    final private static String NETWORK_INPUTS = "networkInputs";
    final private static String NETWORK_OUTPUTS = "networkOutputs";

    final private NetworkIDManager<E> NETWORK_MANAGER; 

    public ProcessInputsRequest(NetworkIDManager<E> idManager) {
        this(idManager, DEFAULT_ENDPOINT);
    }

    public ProcessInputsRequest(NetworkIDManager<E> idManager, String endpoint) {
        super(endpoint,
            //Required Properties
            Map.of(
                NETWORK_INPUTS, PropertyType.mapOf(PropertyType.INTEGER, PropertyType.arrayOf(PropertyType.DOUBLE))
            ),
            //Optional Properties
            NO_PROPERTIES,
            //Response Properties
            Map.of(
                NETWORK_OUTPUTS, PropertyType.mapOf(PropertyType.INTEGER, PropertyType.arrayOf(PropertyType.DOUBLE))
            )
        );
        NETWORK_MANAGER = idManager;
    }

    @Override
    public ResponsePacket handleRequest(JsonObject request) throws Exception {
        RequestHandlerUtils.requireField(request, NETWORK_INPUTS);
        HashMap<Integer, double[]> inputs = RequestHandlerUtils.GSON.fromJson(request.get(NETWORK_INPUTS), new TypeToken<HashMap<Integer, double[]>>(){}.getType());
        Map<Integer, double[]> outputs = NETWORK_MANAGER.processInputs(new Int2ObjectOpenHashMap<>(inputs));
        return ResponsePacket.message(NETWORK_OUTPUTS, outputs);
    }

}

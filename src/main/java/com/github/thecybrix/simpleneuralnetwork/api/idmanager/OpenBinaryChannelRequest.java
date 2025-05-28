package com.github.thecybrix.simpleneuralnetwork.api.idmanager;

import java.util.Map;

import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.server.ContextualJsonRequestHandler;
import com.github.thecybrix.simpleneuralnetwork.server.PropertyType;
import com.github.thecybrix.simpleneuralnetwork.server.ResponsePacket;
import com.google.gson.JsonObject;

public class OpenBinaryChannelRequest<E extends SimpleNeuralNetwork> extends ContextualJsonRequestHandler<NetworkIDManager<E>>{

    final private static String DEFAULT_ENDPOINT = "openBinaryChannel";
    final private static String CHANNEL_PORT = "port";


    public OpenBinaryChannelRequest(NetworkIDManager<E> context) throws NullPointerException {
        this(context, DEFAULT_ENDPOINT);
    }

    public OpenBinaryChannelRequest(NetworkIDManager<E> context, String endpoint) throws NullPointerException {
        super(context, endpoint,
            //Required Properties
            NO_PROPERTIES,
            //Optional Properties
            Map.of(
                CHANNEL_PORT, PropertyType.INTEGER
            ),
            //Response Properties
            NO_PROPERTIES
        );
    }

    @Override
    public ResponsePacket handle(JsonObject request, NetworkIDManager<E> context) throws Exception {
        if(request != null && request.has(CHANNEL_PORT)){
            int port = request.get(CHANNEL_PORT).getAsInt();
            context.openBinaryChannel(port);
        } else {
            context.openBinaryChannel();
        }
        return ResponsePacket.ok();

    }
    
}

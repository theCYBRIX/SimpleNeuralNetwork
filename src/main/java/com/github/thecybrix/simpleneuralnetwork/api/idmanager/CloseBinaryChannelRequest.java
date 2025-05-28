package com.github.thecybrix.simpleneuralnetwork.api.idmanager;

import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.server.ContextualJsonRequestHandler;
import com.github.thecybrix.simpleneuralnetwork.server.ResponsePacket;
import com.google.gson.JsonObject;

public class CloseBinaryChannelRequest<E extends SimpleNeuralNetwork> extends ContextualJsonRequestHandler<NetworkIDManager<E>>{

    final private static String DEFAULT_ENDPOINT = "closeBinaryChannel";


    public CloseBinaryChannelRequest(NetworkIDManager<E> context) throws NullPointerException {
        this(context, DEFAULT_ENDPOINT);
    }

    public CloseBinaryChannelRequest(NetworkIDManager<E> context, String endpoint) throws NullPointerException {
        super(context, endpoint,
            //Required Properties
            NO_PROPERTIES,
            //Optional Properties
            NO_PROPERTIES,
            //Response Properties
            NO_PROPERTIES
        );
    }

    @Override
    public ResponsePacket handle(JsonObject request, NetworkIDManager<E> context) throws Exception {
        if(context.closeBinaryChannel())
            return ResponsePacket.ok();
        else
            return ResponsePacket.error("Binary channel was not open.");
    }
    
}


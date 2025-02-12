package com.github.thecybrix.simpleneuralnetwork.api;

import java.util.Map;
import java.util.Objects;

import com.google.gson.JsonObject;

public abstract class AbstractContextualRequestHandler<E extends APIContext> extends AbstractRequestHandler {

    final private E CONTEXT;

    public AbstractContextualRequestHandler(E context, String endpoint, Map<String, PropertyType> requiredProperties, Map<String, PropertyType> optionalProperties, Map<String, PropertyType> outputProperties) {
        super(endpoint, requiredProperties, optionalProperties, outputProperties);
        CONTEXT = Objects.requireNonNull(context, "Context is null.");
    }
    
    @Override
    final public ResponsePacket handleRequest(JsonObject request) throws Exception {
        return handle(request, CONTEXT);
    }

    public abstract ResponsePacket handle(JsonObject request, E context) throws Exception;
    
}

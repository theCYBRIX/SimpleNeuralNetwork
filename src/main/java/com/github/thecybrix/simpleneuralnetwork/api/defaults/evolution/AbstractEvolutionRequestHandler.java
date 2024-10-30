package com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution;

import java.util.Objects;

import com.github.thecybrix.simpleneuralnetwork.api.RequestHandler;
import com.github.thecybrix.simpleneuralnetwork.api.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.google.gson.JsonObject;

public abstract class AbstractEvolutionRequestHandler<E extends MutableNeuralNetwork> implements RequestHandler{
    final private EvolutionContext<E> CONTEXT;

    private String endpoint;

    public AbstractEvolutionRequestHandler(EvolutionContext<E> context, String endpoint) {
        CONTEXT = Objects.requireNonNull(context, "Context is null.");
        this.endpoint = Objects.requireNonNull(endpoint, "Endpoint is null.");
    }

    @Override
    final public String getKey() {
        return endpoint;
    }

    public void setEndpoint(String endpoint) {
        this.endpoint = endpoint;
    }
    
    @Override
    final public ResponsePacket handle(JsonObject request) throws Exception {
        return handle(request, CONTEXT);
    }

    public abstract ResponsePacket handle(JsonObject request, EvolutionContext<E> context)throws Exception ;
}

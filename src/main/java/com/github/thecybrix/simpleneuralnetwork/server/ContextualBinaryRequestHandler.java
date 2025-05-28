package com.github.thecybrix.simpleneuralnetwork.server;

import java.util.Objects;

import com.github.thecybrix.simpleneuralnetwork.util.EndianAwareInputStream;
import com.github.thecybrix.simpleneuralnetwork.util.EndianAwareOutputStream;

public abstract class ContextualBinaryRequestHandler<E extends APIContext> implements BinaryRequestHandler {
    final private int ENDPOINT;
    final private E CONTEXT;

    public ContextualBinaryRequestHandler(E context, int endpoint) throws NullPointerException {
        CONTEXT = Objects.requireNonNull(context, "Context is null.");
        ENDPOINT = endpoint;
    }
    
    @Override
    final public void handle(EndianAwareInputStream input, EndianAwareOutputStream output) throws Exception {
        handle(input, output, CONTEXT);
    }

    public abstract void handle(EndianAwareInputStream input, EndianAwareOutputStream output, E context) throws Exception;

    @Override
    public int getEndpoint() {
        return ENDPOINT;
    };
    
}

package com.github.thecybrix.simpleneuralnetwork.server;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.Objects;

public abstract class ContextualBinaryRequestHandler<E extends APIContext> implements BinaryRequestHandler {
    final private int ENDPOINT;
    final private E CONTEXT;

    public ContextualBinaryRequestHandler(E context, int endpoint) throws NullPointerException {
        CONTEXT = Objects.requireNonNull(context, "Context is null.");
        ENDPOINT = endpoint;
    }
    
    @Override
    final public void handle(InputStream input, OutputStream output, boolean bigEndian) throws Exception {
        handle(input, output, bigEndian, CONTEXT);
    }

    public abstract void handle(InputStream input, OutputStream output, boolean bigEndian, E context) throws Exception;

    @Override
    public int getEndpoint() {
        return ENDPOINT;
    };
    
}

package com.github.thecybrix.simpleneuralnetwork.server;

import java.util.List;

public interface APIContext {
    public List<JsonRequestHandler> getRequestHandlers();
}

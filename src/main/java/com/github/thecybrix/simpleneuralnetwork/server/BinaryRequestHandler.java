package com.github.thecybrix.simpleneuralnetwork.server;

import com.github.thecybrix.simpleneuralnetwork.util.EndianAwareInputStream;
import com.github.thecybrix.simpleneuralnetwork.util.EndianAwareOutputStream;

public interface BinaryRequestHandler {
    public int getEndpoint();
    public void handle(EndianAwareInputStream input, EndianAwareOutputStream output) throws Exception;
}

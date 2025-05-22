package com.github.thecybrix.simpleneuralnetwork.server;

import java.io.InputStream;
import java.io.OutputStream;

public interface BinaryRequestHandler {
    public int getEndpoint();
    public void handle(InputStream input, OutputStream output, boolean bigEndian) throws Exception;
}

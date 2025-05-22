package com.github.thecybrix.simpleneuralnetwork.server;

import java.io.InputStream;
import java.io.OutputStream;

import com.github.thecybrix.simpleneuralnetwork.util.CallbackInvoker;

public interface IOHandler extends CallbackInvoker<Exception> {

    default void handle(InputStream input, OutputStream output) throws Exception {
        handle(input, output, true);
    }

    public void handle(InputStream input, OutputStream output, boolean bigEndian) throws Exception;

    public void stop();
}

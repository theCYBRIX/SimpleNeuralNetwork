package com.github.thecybrix.simpleneuralnetwork.server;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.logging.Logger;

import com.github.thecybrix.simpleneuralnetwork.util.CallbackInvoker;

public class SimpleStdioServer implements Runnable, CallbackInvoker<SimpleStdioServer>{
    final private static AtomicInteger INSTANCE_COUNTER = new AtomicInteger();

    final private Logger logger;

    final private LinkedList<Consumer<SimpleStdioServer>> CALLBACKS = new LinkedList<>();

    private IOHandler ioHandler;

    public SimpleStdioServer(IOHandler ioHandler){
        this(ioHandler, Integer.toString(INSTANCE_COUNTER.incrementAndGet()));
    }

    public SimpleStdioServer(IOHandler ioHandler, String instanceID){
        this.ioHandler = Objects.requireNonNull(ioHandler, "IOHandler is null.");
        logger = Logger.getLogger(SimpleStdioServer.class.getName() + "-" + Objects.requireNonNull(instanceID, "Instance ID is null."));
    }

    @Override
    public void run() {
        if (System.in == null) throw new IllegalStateException("No InputStream associated with the current JVM.");
        if (System.out == null) throw new IllegalStateException("No OutputStream associated with the current JVM.");

        logger.info("SimpleNNConsole started.");

        try {
            ioHandler.handle(System.in, System.out);
        } catch (Exception e) {
            logError(e);
        }

        logger.info("SimpleNNConsole closed.");
    }

    private void logError(Exception e) {
        logger.warning(e.getMessage());
        logger.fine(stackTraceToString(e));
    }

    public IOHandler getIoHandler() {
        return ioHandler;
    }


    public Logger getLogger() {
        return logger;
    }

    @Override
    public List<Consumer<SimpleStdioServer>> getCallbackList() {
        return CALLBACKS;
    }

    private static String stackTraceToString(Exception e){
        StringWriter stackTrace = new StringWriter();
        e.printStackTrace(new PrintWriter(stackTrace));
        return stackTrace.toString();
    }

}

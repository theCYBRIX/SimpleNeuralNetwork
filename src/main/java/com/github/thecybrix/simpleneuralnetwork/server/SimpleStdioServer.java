package com.github.thecybrix.simpleneuralnetwork.server;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import com.github.thecybrix.simpleneuralnetwork.util.CallbackInvoker;

public class SimpleStdioServer implements Runnable, CallbackInvoker<SimpleStdioServer>{
    final private static Logger LOGGER = Logger.getLogger(SimpleStdioServer.class.getName());

    static {

        try {
            Logger rootLogger = Logger.getLogger("");
            rootLogger.removeHandler(rootLogger.getHandlers()[0]);

            FileHandler logFileHandler = new FileHandler("TestSaves\\SimpleNNConsole.log", false);
            logFileHandler.setFormatter(new SimpleFormatter());
            logFileHandler.setLevel(Level.ALL);
            LOGGER.addHandler(logFileHandler);

            LOGGER.setLevel(Level.ALL);
        } catch (Exception e) {
           LOGGER.severe("Failed to initialize log handler.");
        }
        
    }


    final private LinkedList<Consumer<SimpleStdioServer>> CALLBACKS = new LinkedList<>();

    private IOHandler ioHandler;

    public SimpleStdioServer(IOHandler ioHandler){
        this.ioHandler = Objects.requireNonNull(ioHandler, "IOHandler is null.");
    }

    @Override
    public void run() {
        if (System.in == null) throw new IllegalStateException("No InputStream associated with the current JVM.");
        if (System.out == null) throw new IllegalStateException("No OutputStream associated with the current JVM.");

        LOGGER.info("SimpleNNConsole started.");

        try {
            ioHandler.handle(System.in, System.out);
        } catch (Exception e) {
            logError(e);
        }

        LOGGER.info("SimpleNNConsole closed.");
    }

    private void logError(Exception e) {
        LOGGER.warning(e.getMessage());
        LOGGER.fine(stackTraceToString(e));
    }

    public IOHandler getIoHandler() {
        return ioHandler;
    }

    private static String stackTraceToString(Exception e){
        StringWriter stackTrace = new StringWriter();
        e.printStackTrace(new PrintWriter(stackTrace));
        return stackTrace.toString();
    }

    @Override
    public List<Consumer<SimpleStdioServer>> getCallbackList() {
        return CALLBACKS;
    }

}

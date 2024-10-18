package com.github.thecybrix.simpleneuralnetwork.api;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import com.github.thecybrix.simpleneuralnetwork.api.APIIOHandler.RequestHandler;
import com.github.thecybrix.simpleneuralnetwork.api.APIIOHandler.RequestPacket;
import com.github.thecybrix.simpleneuralnetwork.api.APIIOHandler.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkBuilder;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelector;
import com.github.thecybrix.util.CallbackInvoker;

public class SimpleNNConsole<E extends MutableNeuralNetwork> implements Runnable, CallbackInvoker<SimpleNNConsole<E>>{
    final private static Logger LOGGER = Logger.getLogger(SimpleNNConsole.class.getName());

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


    final private LinkedList<Consumer<SimpleNNConsole<E>>> CALLBACKS = new LinkedList<>();

    private APIIOHandler<E> ioHandler;


    public SimpleNNConsole(NeuralNetworkBuilder<E> networkBuilder) throws IllegalArgumentException, NullPointerException {
        this(networkBuilder, null, null);
    }


    public SimpleNNConsole(NeuralNetworkBuilder<E> networkBuilder, ExecutorService executorService) throws IllegalArgumentException, NullPointerException {
        this(networkBuilder, null, executorService);
    }

    public SimpleNNConsole(NeuralNetworkBuilder<E> networkBuilder, ParentSelector<E> parentSelector) throws IllegalArgumentException, NullPointerException {
        this(networkBuilder, parentSelector, null);
    }

    public SimpleNNConsole(NeuralNetworkBuilder<E> networkBuilder, ParentSelector<E> parentSelector, ExecutorService executorService) throws IllegalArgumentException, NullPointerException {
        networkBuilder = Objects.requireNonNull(networkBuilder, "Network builder is null.");
        executorService = (executorService != null) ? executorService : Executors.newWorkStealingPool();
        parentSelector = (parentSelector != null) ? parentSelector : ParentSelector.eliteSelection();

        ioHandler = new APIIOHandler<>(networkBuilder, parentSelector, executorService);
        ioHandler.addRequestHandler(new RequestHandler<E>() {

            @Override
            public boolean isApplicable(RequestPacket request) {
                return request.getRequest().equalsIgnoreCase("exit");
            }

            @Override
            public ResponsePacket<E> handle(RequestPacket request) throws Exception {
                ioHandler.stop();
                return ResponsePacket.ok();
            }
            
        });
        ioHandler.attachCallback(e -> logError(e));
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

    public APIIOHandler<E> getIoHandler() {
        return ioHandler;
    }

    private static String stackTraceToString(Exception e){
        StringWriter stackTrace = new StringWriter();
        e.printStackTrace(new PrintWriter(stackTrace));
        return stackTrace.toString();
    }

    @Override
    public List<Consumer<SimpleNNConsole<E>>> getCallbackList() {
        return CALLBACKS;
    }

}

package com.github.thecybrix.simpleneuralnetwork.api;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.function.Consumer;
import java.util.logging.ConsoleHandler;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution.EvolutionContext;
import com.github.thecybrix.simpleneuralnetwork.api.defaults.idmanager.NetworkIDManager;
import com.github.thecybrix.simpleneuralnetwork.api.defaults.valuemapping.ValueMappingContext;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkBuilder;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelector;
import com.github.thecybrix.util.CallbackInvoker;
import com.github.thecybrix.util.LELengthPrefixedReader;
import com.github.thecybrix.util.LELengthPrefixedWriter;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

public class APIIOHandler<E extends MutableNeuralNetwork> implements CallbackInvoker<Exception>{
    final private static Logger LOGGER = Logger.getLogger(APIIOHandler.class.getName());

    static {
        class PrintlnFormatter extends Formatter{
            @Override
            public String format(LogRecord record) {
                return record.getMessage() + "\n";
            }
        }

        try {
            Logger rootLogger = Logger.getLogger("");
            rootLogger.removeHandler(rootLogger.getHandlers()[0]);

            FileHandler logFileHandler = new FileHandler("TestSaves\\APIIOHandler.log", false);
            logFileHandler.setFormatter(new SimpleFormatter());
            logFileHandler.setLevel(Level.ALL);
            LOGGER.addHandler(logFileHandler);

            ConsoleHandler consoleHandler = new ConsoleHandler();
            consoleHandler.setFormatter(new PrintlnFormatter());
            consoleHandler.setLevel(Level.INFO);
            LOGGER.addHandler(consoleHandler);

            LOGGER.setLevel(Level.ALL);
        } catch (Exception e) {
           LOGGER.severe("Failed to initialize log handler.");
        }
        
    }

    final public static String[] REQUEST_FIELDS = new String[]{
        "request",
        "payload"
    };

    final private LinkedList<Consumer<Exception>> CALLBACKS = new LinkedList<>();
    final private HashMap<String, RequestHandler> REQUEST_HANDLERS = new HashMap<>();

    
    private volatile boolean keepAlive = false;


    public APIIOHandler(NeuralNetworkBuilder<E> networkBuilder) throws IllegalArgumentException, NullPointerException {
        this(networkBuilder, null, null);
    }

    public APIIOHandler(NeuralNetworkBuilder<E> networkBuilder, ExecutorService executorService) throws IllegalArgumentException, NullPointerException {
        this(networkBuilder, null, executorService);
    }

    public APIIOHandler(NeuralNetworkBuilder<E> networkBuilder, ParentSelector<E> parentSelector) throws IllegalArgumentException, NullPointerException {
        this(networkBuilder, parentSelector, null);
    }

    public APIIOHandler(NeuralNetworkBuilder<E> networkBuilder, ParentSelector<E> parentSelector, ExecutorService executorService) throws IllegalArgumentException, NullPointerException {
        NetworkIDManager<E> newtorkIdManager = new NetworkIDManager<>(executorService);
        EvolutionContext<E> evolutionContext = new EvolutionContext<>(newtorkIdManager, networkBuilder, parentSelector);
        ValueMappingContext<E> valueMappingContextContext = new ValueMappingContext<>(newtorkIdManager, networkBuilder, parentSelector);
        addRequestHandlers(newtorkIdManager.getRequestHandlers());
        addRequestHandlers(evolutionContext.getRequestHandlers());
        addRequestHandlers(valueMappingContextContext.getRequestHandlers());

        addRequestHandler(new EndpointsRequest(this));
    }


    public void handle(InputStream input, OutputStream output) throws InterruptedException {
        try(
            LELengthPrefixedReader reader = new LELengthPrefixedReader(input);
            LELengthPrefixedWriter writer = new LELengthPrefixedWriter(output);
        ) {
            keepAlive = true;
            while(keepAlive){
                if(Thread.interrupted()) throw new InterruptedException();

                String request = reader.readString();
                LOGGER.finest(() -> "Request received:\n" + request);

                String response = RequestHandlerUtils.GSON.toJson(handleRequest(request));
                LOGGER.finest(() -> "Response packet:\n" + response);

                writer.writeString(response);
                writer.flush();
            }
        } catch (Exception e) {
            logError(e);
        }
    }

    public void stop(){
        keepAlive = false;
    }

    private void logError(Exception e){
        LOGGER.warning(e.getMessage());
        LOGGER.fine(RequestHandlerUtils.stackTraceToString(e));
        processCallbacks(e);
    }

    private ResponsePacket handleRequest(String request) {
        try {
            JsonObject r = JsonParser.parseString(request).getAsJsonObject();
            RequestHandlerUtils.requireField(r, REQUEST_FIELDS[0]);
                        
            String endpoint = r.get(REQUEST_FIELDS[0]).getAsString();
            RequestHandler handler = REQUEST_HANDLERS.get(endpoint);

            if(handler == null)
                return ResponsePacket.error("Invalid request.", "\"" + endpoint + "\" is not a recognized command.");

            JsonObject data = r.getAsJsonObject(REQUEST_FIELDS[1]);

            return handler.handle(data);
            
        } catch (Exception e) {
            LOGGER.warning("Failed to handle request: " + e.getClass().getSimpleName());
            logError(e);
            return ResponsePacket.error(e.getClass().getSimpleName(), e.getMessage(), RequestHandlerUtils.stackTraceToString(e));
        }
    }

    public int getRequestHandlerCount(){
        return REQUEST_HANDLERS.size();
    }

    public Collection<RequestHandler> getRequestHandlers(){
        return REQUEST_HANDLERS.values();
    }

    public Optional<RequestHandler> getRequestHandler(String endpooint){
        return Optional.ofNullable(REQUEST_HANDLERS.get(endpooint));
    }
    
    public void addRequestHandler(RequestHandler handler) throws NullPointerException {
        if(handler == null) throw new NullPointerException("Request handler is null.");
        REQUEST_HANDLERS.put(handler.getEndpoint(), handler);
    }

    public void addRequestHandlers(List<RequestHandler> handlers) throws NullPointerException {
        if(handlers == null) throw new NullPointerException("Request handlers is null.");
        handlers.parallelStream().filter(x -> x != null);
        for (RequestHandler handler : handlers)
            REQUEST_HANDLERS.put(handler.getEndpoint(), handler);
    }
    
    public boolean removeRequestHandler(RequestHandler handler){
        if(REQUEST_HANDLERS.remove(handler.getEndpoint(), handler)) return true;
        if(REQUEST_HANDLERS.containsValue(handler))
            return REQUEST_HANDLERS.values().remove(handler);
        return false;
    }
    
    public RequestHandler removeRequestHandler(String endpoint){
        return REQUEST_HANDLERS.remove(endpoint);
    }

    @Override
    public List<Consumer<Exception>> getCallbackList() {
        return CALLBACKS;
    }
    
}

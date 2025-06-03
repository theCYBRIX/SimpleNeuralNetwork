package com.github.thecybrix.simpleneuralnetwork.server;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.SocketException;
import java.text.DecimalFormat;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;
import com.github.thecybrix.simpleneuralnetwork.exceptions.EndpointConflictException;
import com.github.thecybrix.simpleneuralnetwork.util.LengthPrefixedReader;
import com.github.thecybrix.simpleneuralnetwork.util.LengthPrefixedWriter;
import com.github.thecybrix.simpleneuralnetwork.util.StopWatch;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

public class JsonIOHandler implements IOHandler {
    final private static AtomicInteger INSTANCE_COUNTER = new AtomicInteger();

    final public static String[] REQUEST_FIELDS = new String[]{
        "request",
        "payload"
    };

    final private Logger logger;
    final private LinkedList<Consumer<Exception>> CALLBACKS = new LinkedList<>();
    final private HashMap<String, JsonRequestHandler> REQUEST_HANDLERS = new HashMap<>();
    final private DecimalFormat loggingDecimalFormat = new DecimalFormat("#.####");

    
    private StopWatch executionTimer;
    private StopWatch dialogTimer;
    private boolean logExecutionTimes = false;
    private HandleFunction handleFunction = this::handleDirect;

    private volatile boolean keepAlive = false;

    public JsonIOHandler() {
        this(Integer.toString(INSTANCE_COUNTER.incrementAndGet()));
    }

    public JsonIOHandler(String instanceID) {
        addRequestHandler(new EndpointsRequest(this));
        logger = Logger.getLogger(JsonIOHandler.class.getName() + "-" + Objects.requireNonNull(instanceID, "Instance ID is null."));
    }

    @Override
    public void handle(InputStream input, OutputStream output, boolean bigEndian) throws InterruptedException {
        try(
            LengthPrefixedReader reader = new LengthPrefixedReader(input, bigEndian);
            LengthPrefixedWriter writer = new LengthPrefixedWriter(output, bigEndian);
        ) {
            keepAlive = true;
            while(keepAlive){
                if(Thread.interrupted()) throw new InterruptedException();

                int requestLength = reader.readLengthPrefix();

                handleFunction.handle(reader, writer, requestLength);
            }
        } catch (Exception e) {
            logError(e);
        }
    }

    private void handleDirect(LengthPrefixedReader reader, LengthPrefixedWriter writer, int length) throws SocketException, IOException{
        String request = reader.readString(length);

        if(logger.isLoggable(Level.FINEST))
            logger.finest("Request received:\n" + request);
        
        ResponsePacket responsePacket = handleRequest(request);

        String response = RequestHandlerUtils.GSON.toJson(responsePacket);

        writer.writeString(response);
        writer.flush();

        if(logger.isLoggable(Level.FINEST))
            logger.finest("Response packet:\n" + response);

    }

    private void handleTimed(LengthPrefixedReader reader, LengthPrefixedWriter writer, int length) throws SocketException, IOException{

        boolean timersActive = logExecutionTimes && logger.isLoggable(Level.FINE);
        if(timersActive){
            dialogTimer.start();
            executionTimer.start();
        }

        String request = reader.readString(length);

        if(timersActive){
            executionTimer.stop();
            logger.fine("Reading: " + loggingDecimalFormat.format(executionTimer.getMillisExact()) + "ms");
        }

        if(logger.isLoggable(Level.FINEST))
            logger.finest("Request received:\n" + request);

        if(timersActive){
            executionTimer.start();
        }
        ResponsePacket responsePacket = handleRequest(request);
        if(timersActive){
            executionTimer.stop();
            logger.fine("Processing: " + loggingDecimalFormat.format(executionTimer.getMillisExact()) + "ms");
            executionTimer.start();
        }
        
        String response = RequestHandlerUtils.GSON.toJson(responsePacket);
        if(timersActive){
            executionTimer.stop();
            logger.fine("Serializing: " + loggingDecimalFormat.format(executionTimer.getMillisExact()) + "ms");
            executionTimer.start();
        }

        writer.writeString(response);
        writer.flush();

        if(timersActive){
            executionTimer.stop();
            dialogTimer.stop();
            logger.fine("Sending: " + loggingDecimalFormat.format(executionTimer.getMillisExact()) + "ms");
            logger.fine("Total Handle Time: " + loggingDecimalFormat.format(dialogTimer.getMillisExact()) + "ms");
        }

        if(logger.isLoggable(Level.FINEST))
            logger.finest("Response sent:\n" + response);
        
    }

    public void stop(){
        keepAlive = false;
    }

    public void setDialogTimeLogging(boolean enabled){
        this.handleFunction = enabled ? this::handleTimed : this::handleDirect;
    }

    private void logError(Exception e){
        logger.warning(e.getMessage());
        logger.fine(RequestHandlerUtils.stackTraceToString(e));
        processCallbacks(e);
    }


    private ResponsePacket handleRequest(String request) {
        try {
            JsonObject r = JsonParser.parseString(request).getAsJsonObject();
            RequestHandlerUtils.requireField(r, REQUEST_FIELDS[0]);
                        
            String endpoint = r.get(REQUEST_FIELDS[0]).getAsString();
            JsonRequestHandler handler = REQUEST_HANDLERS.get(endpoint);

            if(handler == null)
                return ResponsePacket.error("Invalid request.", "\"" + endpoint + "\" is not a recognized command.");

            JsonObject data = r.getAsJsonObject(REQUEST_FIELDS[1]);

            return handler.handle(data);
            
        } catch (Exception e) {
            logger.warning("Failed to handle request: " + e.getClass().getSimpleName());
            logError(e);
            return ResponsePacket.error(e.getClass().getSimpleName(), e.getMessage(), RequestHandlerUtils.stackTraceToString(e));
        }
    }
    
    public void setLogExecutionTimes(boolean enabled) {
        if(enabled){
            dialogTimer = new StopWatch();
            executionTimer = new StopWatch();
        } else {
            dialogTimer = null;
            executionTimer = null;
        }
        this.logExecutionTimes = enabled;
    }

    public int getRequestHandlerCount(){
        return REQUEST_HANDLERS.size();
    }

    public Collection<JsonRequestHandler> getRequestHandlers(){
        return REQUEST_HANDLERS.values();
    }

    public Optional<JsonRequestHandler> getRequestHandler(String endpoint){
        return Optional.ofNullable(REQUEST_HANDLERS.get(endpoint));
    }
    
    public void addRequestHandler(JsonRequestHandler handler) throws EndpointConflictException, NullPointerException {
        if(handler == null) throw new NullPointerException("Request handler is null.");
        if(REQUEST_HANDLERS.containsKey(handler.getEndpoint())) throw new EndpointConflictException("Endpoint \"" + handler.getEndpoint() + "\" has already been defined.");
        REQUEST_HANDLERS.put(handler.getEndpoint(), handler);
    }

    public void addRequestHandlers(Collection<JsonRequestHandler> handlers) throws EndpointConflictException, NullPointerException {
        if(handlers == null) throw new NullPointerException("Request handlers list is null.");
        if(handlers.parallelStream().anyMatch(x -> x == null)) throw new NullPointerException("Request handlers list contains null.");

        HashMap<String, Integer> endpointCounts = new HashMap<>();
        for(JsonRequestHandler handler : handlers){
            String endpoint = handler.getEndpoint();
            if(endpointCounts.containsKey(endpoint))
                endpointCounts.put(endpoint, endpointCounts.get(endpoint) + 1);
            else
                endpointCounts.put(endpoint, 1);
        }
        HashMap<String, Integer> conflicts = new HashMap<>();
        for(String endpoint : endpointCounts.keySet()){
            Integer count = endpointCounts.get(endpoint);
            if(count.intValue() > 1)
                conflicts.put(endpoint, count);
        }
        if(conflicts.size() > 0){
            StringBuilder errorMsg = new StringBuilder("Request handlers list contains endpoint conflict");
            errorMsg.append(conflicts.size() > 1 ? "s:" : ":");
            for(String endpoint : conflicts.keySet()){
                errorMsg.append("\n\"")
                        .append(endpoint)
                        .append("\" (defined ")
                        .append(conflicts.get(endpoint).intValue())
                        .append(" times)");
            }
            throw new EndpointConflictException(errorMsg.toString());
        }
            
        for (JsonRequestHandler handler : handlers)
            REQUEST_HANDLERS.put(handler.getEndpoint(), handler);
    }
    
    public boolean removeRequestHandler(JsonRequestHandler handler) throws NullPointerException{
        if(REQUEST_HANDLERS.remove(Objects.requireNonNull(handler, "Handler is null.").getEndpoint(), handler)) return true;
        if(REQUEST_HANDLERS.containsValue(handler))
            return REQUEST_HANDLERS.values().remove(handler);
        return false;
    }
    
    public JsonRequestHandler removeRequestHandler(String endpoint){
        return REQUEST_HANDLERS.remove(endpoint);
    }
    
    public void removeRequestHandlers(Collection<JsonRequestHandler> handlers){
        for (JsonRequestHandler jsonRequestHandler : handlers)
            removeRequestHandler(jsonRequestHandler);
    }

    public Logger getLogger() {
        return logger;
    }

    @Override
    public List<Consumer<Exception>> getCallbackList() {
        return CALLBACKS;
    }

    @FunctionalInterface
    private static interface HandleFunction {
        public void handle(LengthPrefixedReader reader, LengthPrefixedWriter writer, int length) throws SocketException, IOException;
    }
    
}

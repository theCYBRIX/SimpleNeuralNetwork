package com.github.thecybrix.simpleneuralnetwork.server;

import java.io.InputStream;
import java.io.OutputStream;
import java.text.DecimalFormat;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.logging.ConsoleHandler;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import com.github.thecybrix.simpleneuralnetwork.exceptions.EndpointConflictException;
import com.github.thecybrix.simpleneuralnetwork.util.LengthPrefixedReader;
import com.github.thecybrix.simpleneuralnetwork.util.LengthPrefixedWriter;
import com.github.thecybrix.simpleneuralnetwork.util.StopWatch;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

public class JsonIOHandler implements IOHandler {
    final private static Logger LOGGER = Logger.getLogger(JsonIOHandler.class.getName());

    static {
        class PrintlnFormatter extends Formatter{
            @Override
            public String format(LogRecord record) {
                return record.getMessage() + "\n";
            }
        }

        try {
            FileHandler logFileHandler = new FileHandler("TestSaves\\APIIOHandler.log", false);
            logFileHandler.setFormatter(new SimpleFormatter());
            logFileHandler.setLevel(Level.ALL);
            LOGGER.addHandler(logFileHandler);

            ConsoleHandler consoleHandler = new ConsoleHandler();
            consoleHandler.setFormatter(new PrintlnFormatter());
            consoleHandler.setLevel(Level.ALL);
            LOGGER.addHandler(consoleHandler);

            LOGGER.setLevel(Level.INFO);
        } catch (Exception e) {
            LOGGER.severe("Failed to initialize log handler.");
        }
        
    }

    final public static String[] REQUEST_FIELDS = new String[]{
        "request",
        "payload"
    };

    final private LinkedList<Consumer<Exception>> CALLBACKS = new LinkedList<>();
    final private HashMap<String, JsonRequestHandler> REQUEST_HANDLERS = new HashMap<>();
    final private DecimalFormat loggingDecimalFormat = new DecimalFormat("#.####");

    
    private StopWatch executionTimer;
    private StopWatch dialogTimer;
    private boolean logExecutionTimes = false;

    private volatile boolean keepAlive = false;

    public JsonIOHandler() {
        addRequestHandler(new EndpointsRequest(this));
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

                boolean timersActive = logExecutionTimes && LOGGER.isLoggable(Level.INFO);
                if(timersActive){
                    dialogTimer.start();
                    executionTimer.start();
                }

                String request = reader.readString(requestLength);

                if(timersActive){
                    executionTimer.stop();
                    LOGGER.info("Reading: " + loggingDecimalFormat.format(executionTimer.getMillisExact()) + "ms");
                }

                if(LOGGER.isLoggable(Level.FINEST))
                    LOGGER.finest("Request received:\n" + request);

                if(timersActive){
                    executionTimer.start();
                }
                ResponsePacket responsePacket = handleRequest(request);
                if(timersActive){
                    executionTimer.stop();
                    LOGGER.info("Processing: " + loggingDecimalFormat.format(executionTimer.getMillisExact()) + "ms");
                    executionTimer.start();
                }
                
                String response = RequestHandlerUtils.GSON.toJson(responsePacket);
                if(timersActive){
                    executionTimer.stop();
                    LOGGER.info("Serializing: " + loggingDecimalFormat.format(executionTimer.getMillisExact()) + "ms");
                    executionTimer.start();
                }

                writer.writeString(response);
                writer.flush();

                if(timersActive){
                    executionTimer.stop();
                    dialogTimer.stop();
                    LOGGER.info("Sending: " + loggingDecimalFormat.format(executionTimer.getMillisExact()) + "ms");
                    LOGGER.info("Total Handle Time: " + loggingDecimalFormat.format(dialogTimer.getMillisExact()) + "ms");
                }

                if(LOGGER.isLoggable(Level.FINEST))
                    LOGGER.finest("Response packet:\n" + response);
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
            JsonRequestHandler handler = REQUEST_HANDLERS.get(endpoint);

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

    public void addRequestHandlers(List<JsonRequestHandler> handlers) throws EndpointConflictException, NullPointerException {
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
    
    public boolean removeRequestHandler(JsonRequestHandler handler){
        if(REQUEST_HANDLERS.remove(handler.getEndpoint(), handler)) return true;
        if(REQUEST_HANDLERS.containsValue(handler))
            return REQUEST_HANDLERS.values().remove(handler);
        return false;
    }
    
    public JsonRequestHandler removeRequestHandler(String endpoint){
        return REQUEST_HANDLERS.remove(endpoint);
    }

    @Override
    public List<Consumer<Exception>> getCallbackList() {
        return CALLBACKS;
    }
    
}

package com.github.thecybrix.simpleneuralnetwork.server;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
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
import com.github.thecybrix.simpleneuralnetwork.util.EndianAwareInputStream;
import com.github.thecybrix.simpleneuralnetwork.util.EndianAwareOutputStream;
import com.github.thecybrix.simpleneuralnetwork.util.Endianness;

import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;

public class BinaryIOHandler implements IOHandler {
    final private static Logger LOGGER = Logger.getLogger(JsonIOHandler.class.getName());

    final public static byte ERR_UNDEFINED_ENDPOINT = 1;

    static {
        class PrintlnFormatter extends Formatter{
            @Override
            public String format(LogRecord record) {
                return record.getMessage() + "\n";
            }
        }

        try {
            FileHandler logFileHandler = new FileHandler("TestSaves\\BinaryIOHandler.log", false);
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

    final private ArrayList<Consumer<Exception>> CALLBACKS = new ArrayList<>(0);

    final private Int2ObjectOpenHashMap<BinaryRequestHandler> REQUEST_HANDLERS = new Int2ObjectOpenHashMap<>();

    private volatile boolean keepAlive = false;

    public int getRequestHandlerCount(){
        return REQUEST_HANDLERS.size();
    }

    public Collection<BinaryRequestHandler> getRequestHandlers(){
        return REQUEST_HANDLERS.values();
    }

    public Optional<BinaryRequestHandler> getRequestHandler(int endpoint){
        return Optional.ofNullable(REQUEST_HANDLERS.get(endpoint));
    }

    public void handle(InputStream input, OutputStream output) throws InterruptedException {
        handle(input, output, true);
    }

    public void handle(InputStream input, OutputStream output, boolean bigEndian) throws InterruptedException {
        Endianness endian = bigEndian ? Endianness.BIG_ENDIAN : Endianness.LITTLE_ENDIAN;
        try (
            EndianAwareInputStream bufferedInput = new EndianAwareInputStream(new BufferedInputStream(input), endian);
            EndianAwareOutputStream bufferedOutput = new EndianAwareOutputStream(new BufferedOutputStream(output), endian); 
        ) {
            keepAlive = true;
            while(keepAlive){
                if(Thread.interrupted()) throw new InterruptedException(); 

                int request = bufferedInput.readInt();

                BinaryRequestHandler handler = REQUEST_HANDLERS.get(request);

                if(handler == null){
                    bufferedOutput.writeByte((byte)-1);
                    bufferedOutput.writeInt(ERR_UNDEFINED_ENDPOINT);
                } else {
                    handler.handle(bufferedInput, bufferedOutput);
                }

                bufferedOutput.flush();
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
    
    public void addRequestHandler(BinaryRequestHandler handler) throws EndpointConflictException, NullPointerException {
        if(handler == null) throw new NullPointerException("Request handler is null.");
        if(REQUEST_HANDLERS.containsKey(handler.getEndpoint())) throw new EndpointConflictException("Endpoint \"" + handler.getEndpoint() + "\" has already been defined.");
        REQUEST_HANDLERS.put(handler.getEndpoint(), handler);
    }

    public void addRequestHandlers(List<BinaryRequestHandler> handlers) throws EndpointConflictException, NullPointerException {
        if(handlers == null) throw new NullPointerException("Request handlers list is null.");
        if(handlers.parallelStream().anyMatch(x -> x == null)) throw new NullPointerException("Request handlers list contains null.");

        HashMap<Integer, Integer> endpointCounts = new HashMap<>();
        for(BinaryRequestHandler handler : handlers){
            int endpoint = handler.getEndpoint();
            if(endpointCounts.containsKey(endpoint))
                endpointCounts.put(endpoint, endpointCounts.get(endpoint) + 1);
            else
                endpointCounts.put(endpoint, 1);
        }
        HashMap<Integer, Integer> conflicts = new HashMap<>();
        for(Integer endpoint : endpointCounts.keySet()){
            Integer count = endpointCounts.get(endpoint);
            if(count.intValue() > 1)
                conflicts.put(endpoint, count);
        }
        if(conflicts.size() > 0){
            StringBuilder errorMsg = new StringBuilder("Request handlers list contains endpoint conflict");
            errorMsg.append(conflicts.size() > 1 ? "s:" : ":");
            for(Integer endpoint : conflicts.keySet()){
                errorMsg.append("\n\"")
                        .append(endpoint)
                        .append("\" (defined ")
                        .append(conflicts.get(endpoint).intValue())
                        .append(" times)");
            }
            throw new EndpointConflictException(errorMsg.toString());
        }
            
        for (BinaryRequestHandler handler : handlers)
            REQUEST_HANDLERS.put(handler.getEndpoint(), handler);
    }
    
    public boolean removeRequestHandler(BinaryRequestHandler handler){
        if(REQUEST_HANDLERS.remove(handler.getEndpoint(), handler)) return true;
        if(REQUEST_HANDLERS.containsValue(handler))
            return REQUEST_HANDLERS.values().remove(handler);
        return false;
    }
    
    public BinaryRequestHandler removeRequestHandler(int endpoint){
        return REQUEST_HANDLERS.remove(endpoint);
    }

    @Override
    public List<Consumer<Exception>> getCallbackList() {
        return CALLBACKS;
    }
    
}

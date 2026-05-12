package com.github.thecybrix.simpleneuralnetwork.server;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.logging.Logger;

import com.github.thecybrix.simpleneuralnetwork.server.raw.RawRequest;
import com.github.thecybrix.simpleneuralnetwork.server.raw.RawRequestDispatcher;
import com.github.thecybrix.simpleneuralnetwork.server.raw.RawResponse;
import com.github.thecybrix.simpleneuralnetwork.util.EndianAwareInputStream;
import com.github.thecybrix.simpleneuralnetwork.util.EndianAwareOutputStream;
import com.github.thecybrix.simpleneuralnetwork.util.Endianness;

public class SimpleIOHandler implements IOHandler {
    final private static Logger logger;
    final private RawRequestDispatcher DISPATCHER;

    static {
        logger = Logger.getLogger(SimpleIOHandler.class.getName());
    }

    public SimpleIOHandler(RawRequestDispatcher dispatcher) {
        this(dispatcher, Endianness.BIG_ENDIAN);
    }

    public SimpleIOHandler(RawRequestDispatcher dispatcher, Endianness endianness) {
        DISPATCHER = Objects.requireNonNull(dispatcher, "Dispatcher is null.");
        this.endianness = endianness;
    }

    private LinkedList<Consumer<Exception>> callbacks = new LinkedList<>();

    // private HashMap<String, SessionContext> sessions = new HashMap<>();
    private Endianness endianness = Endianness.BIG_ENDIAN;

    private boolean keepRunning;
    private Thread processThread;
    

    @Override
    public void handle(InputStream input, OutputStream output) throws Exception {
        keepRunning = true;
        try (
            EndianAwareInputStream inputStream = new EndianAwareInputStream(new BufferedInputStream(input), endianness);
            EndianAwareOutputStream outputStream = new EndianAwareOutputStream(new BufferedOutputStream(output), endianness)
        ) {
            //TODO: Keep track of sessions
            LinkedBlockingQueue<RawResponse> pendingResponses = new LinkedBlockingQueue<>();

            Thread writerThread = new Thread(() -> writerLoop(outputStream, pendingResponses, e -> { logger.warning(e::toString); return false; }), "Writer");
            writerThread.start();

            try {
                readerLoop(inputStream, pendingResponses);
            } finally {
                writerThread.interrupt();
            }
            
        } catch (IOException e) {
            if (!(e instanceof EOFException) || keepRunning) throw e;
        }// } catch (InterruptedException e) {
        //     if (keepRunning) throw e;
        // }
    }

    private void readerLoop(EndianAwareInputStream inputStream, BlockingQueue<RawResponse> pendingResponses) throws IOException {
        SessionContext context = new SessionContext();
        while (keepRunning) {
            RawRequest request = readRequest(inputStream);
            CompletableFuture<RawResponse> future = DISPATCHER.dispatch(context, request);
            future.whenComplete(
                (response, exception) -> {
                    if(response != null) {
                        pendingResponses.add(response);
                    } else {
                        pendingResponses.add(RawResponse.respondingTo(request, -1, exception != null ? exception.getMessage().getBytes() : new byte[0]));
                    }
                }
            );
        }
    }

    private void writerLoop(EndianAwareOutputStream outputStream, BlockingQueue<RawResponse> pendingWrites, Function<Exception, Boolean> onUnhandledException) {
        while (keepRunning) {
            try {
                RawResponse response = pendingWrites.take();
                writeResponse(outputStream, response);
            } catch (EOFException | InterruptedException e) {
                break;
            } catch (IOException e){
                boolean fatal = onUnhandledException.apply(e);
                if (fatal) break;
            }
        }
    }

    private RawRequest readRequest(EndianAwareInputStream inputStream) throws IOException {
        RawRequest request = new RawRequest();
        request.encodingType = inputStream.readInt();
        request.requestID = inputStream.readInt();
        
        int length = inputStream.readInt();
        request.payload = inputStream.readNBytes(length);

        return request;
    }

    private void writeResponse(EndianAwareOutputStream outputStream, RawResponse response) throws IOException {
        outputStream.writeInt(response.encodingType);
        outputStream.writeInt(response.requestID);
        outputStream.writeInt(response.payload.length);
        outputStream.writeNBytes(response.payload);
        outputStream.flush();
    }

    @Override
    public void stop() {
        keepRunning = false;
        if (processThread != null) {
            processThread.interrupt();
        }
    }

    public Endianness getEndianness() {
        return endianness;
    }

    public void setEndianness(Endianness endianness) {
        this.endianness = endianness;
    }

    @Override
    public List<Consumer<Exception>> getCallbackList() {
        return callbacks;
    }
    
}

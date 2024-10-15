package com.github.thecybrix.simpleneuralnetwork.api;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.Inet4Address;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;
import java.util.logging.ConsoleHandler;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkBuilder;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelector;
import com.github.thecybrix.util.AutoRunnable;
import com.github.thecybrix.util.CallbackInvoker;

public class SimpleNNServer<E extends MutableNeuralNetwork> implements AutoRunnable, CallbackInvoker<SimpleNNServer<E>> {
    final private static Logger LOGGER = Logger.getLogger(SimpleNNServer.class.getName());

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

            FileHandler logFileHandler = new FileHandler("TestSaves\\SimpleNNServer.log", false);
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

    final private LinkedList<Consumer<SimpleNNServer<E>>> CALLBACKS = new LinkedList<>();

    private APIIOHandler<E> ioHandler;

    private ServerSocket serverSocket;
    private Socket client;
    private boolean running = false;
    private Thread runningThread = null;

    private int serverPort;
    private volatile boolean closeRequested = false; 

    public SimpleNNServer(int serverPort, NeuralNetworkBuilder<E> networkBuilder) throws IllegalArgumentException, NullPointerException {
        this(serverPort, networkBuilder, null, null);
    }


    public SimpleNNServer(int serverPort, NeuralNetworkBuilder<E> networkBuilder, ExecutorService executorService) throws IllegalArgumentException, NullPointerException {
        this(serverPort, networkBuilder, null, executorService);
    }

    public SimpleNNServer(int serverPort, NeuralNetworkBuilder<E> networkBuilder, ParentSelector<E> parentSelector) throws IllegalArgumentException, NullPointerException {
        this(serverPort, networkBuilder, parentSelector, null);
    }

    public SimpleNNServer(int serverPort, NeuralNetworkBuilder<E> networkBuilder, ParentSelector<E> parentSelector, ExecutorService executorService) throws IllegalArgumentException, NullPointerException {
        networkBuilder = Objects.requireNonNull(networkBuilder, "Network builder is null.");
        setPort(serverPort);
        executorService = (executorService != null) ? executorService : Executors.newWorkStealingPool();
        parentSelector = (parentSelector != null) ? parentSelector : ParentSelector.eliteSelection();

        ioHandler = new APIIOHandler<>(networkBuilder, parentSelector, executorService);
        ioHandler.attachCallback(e -> { if(!(e instanceof SocketException)) logError(e); });
    }

    @Override
    public void run() {
        closeRequested = false;

        try (ServerSocket socket = new ServerSocket(serverPort)) {
            runningThread = Thread.currentThread();
            serverSocket = socket;
            running = true;

            LOGGER.info("Server started.\nAddress: " + Inet4Address.getLocalHost().getHostAddress() + "\nPort: " + socket.getLocalPort() + "\n");

            while(!closeRequested){
                LOGGER.info("Waiting on connection...");

                client = socket.accept();

                LOGGER.info("Client connected: " + client.getInetAddress());

                ioHandler.handle(client.getInputStream(), client.getOutputStream());

                LOGGER.info("Client disconnected: " + client.getInetAddress());
                client = null;
            }
        } catch (SocketException e) {
            if(!closeRequested) logError(e);;
        } catch (Exception e) {
            logError(e);
        } finally {
            running = false;
            runningThread = null;
            serverSocket = null;
            LOGGER.info("Server shutdown.");
        }
    }

    private void logError(Exception e){
        LOGGER.warning(e.getMessage());
        LOGGER.fine(stackTraceToString(e));
    }

    public APIIOHandler<E> getIoHandler() {
        return ioHandler;
    }

    private void setPort(int port) throws IllegalArgumentException {
        if(port < 0 || port > 65535) throw new IllegalArgumentException("Illegal port number requested: \"" + port + "\".\nPort must be in range [0, 65535].");
        this.serverPort = port;
    }

    @Override
    public void stop() throws SecurityException {
        if(running) {
            closeRequested = true;
            runningThread.interrupt();
            try {
                serverSocket.close();
                if (client != null) client.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public boolean isRunning() {
        return running;
    }

    @Override
    public Optional<Thread> getRunningThread() {
        return Optional.ofNullable(runningThread);
    }

    private static String stackTraceToString(Exception e){
        StringWriter stackTrace = new StringWriter();
        e.printStackTrace(new PrintWriter(stackTrace));
        return stackTrace.toString();
    }

    @Override
    public List<Consumer<SimpleNNServer<E>>> getCallbackList() {
        return CALLBACKS;
    }
    
}

package com.github.thecybrix.simpleneuralnetwork.server;

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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.logging.Logger;
import com.github.thecybrix.simpleneuralnetwork.api.idmanager.NetworkIDManager;
import com.github.thecybrix.simpleneuralnetwork.util.AutoRunnable;
import com.github.thecybrix.simpleneuralnetwork.util.CallbackInvoker;

public class SimpleTCPServer implements AutoRunnable, CallbackInvoker<SimpleTCPServer> {
    final private static AtomicInteger INSTANCE_COUNTER = new AtomicInteger();

    final private Logger logger;
    final private LinkedList<Consumer<SimpleTCPServer>> CALLBACKS = new LinkedList<>();


    private IOHandler ioHandler;

    private ServerSocket serverSocket;
    private Socket client;
    private boolean running = false;
    private Thread runningThread = null;

    private int serverPort;
    private volatile boolean closeRequested = false;

    public SimpleTCPServer(int serverPort, IOHandler ioHandler) throws IllegalArgumentException, NullPointerException {
        this(serverPort, ioHandler, Integer.toString(INSTANCE_COUNTER.incrementAndGet()));
    }

    public SimpleTCPServer(int serverPort, IOHandler ioHandler, String instanceID) throws IllegalArgumentException, NullPointerException {
        setPort(serverPort);
        this.ioHandler = Objects.requireNonNull(ioHandler, "IOHandler is null.");
        this.ioHandler.attachCallback(e -> { if(!(e instanceof SocketException)) logError(e); });
        logger = Logger.getLogger(NetworkIDManager.class.getName() + "-" + Objects.requireNonNull(instanceID, "Instance ID is null."));
    }

    @Override
    public void run() {
        closeRequested = false;

        try (ServerSocket socket = new ServerSocket(serverPort)) {
            runningThread = Thread.currentThread();
            serverSocket = socket;
            running = true;

            logger.info("Server started.\nAddress: " + Inet4Address.getLocalHost().getHostAddress() + "\nPort: " + socket.getLocalPort() + "\n");

            while(!closeRequested){
                logger.info("Waiting on connection...");

                client = socket.accept();

                logger.info("Client connected: " + client.getInetAddress());

                ioHandler.handle(client.getInputStream(), client.getOutputStream());

                logger.info("Client disconnected: " + client.getInetAddress());
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
            logger.info("Server shutdown.");
        }
    }

    private void logError(Exception e){
        logger.warning(e.getMessage());
        logger.fine(stackTraceToString(e));
    }

    public IOHandler getIoHandler() {
        return ioHandler;
    }

    public Logger getLogger() {
        return logger;
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

    @Override
    public List<Consumer<SimpleTCPServer>> getCallbackList() {
        return CALLBACKS;
    }

    private static String stackTraceToString(Exception e){
        StringWriter stackTrace = new StringWriter();
        e.printStackTrace(new PrintWriter(stackTrace));
        return stackTrace.toString();
    }
    
}

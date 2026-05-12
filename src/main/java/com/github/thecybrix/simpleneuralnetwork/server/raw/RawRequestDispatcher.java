package com.github.thecybrix.simpleneuralnetwork.server.raw;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Consumer;

import com.github.thecybrix.simpleneuralnetwork.exceptions.EndpointConflictException;
import com.github.thecybrix.simpleneuralnetwork.exceptions.NoSuchRequestTypeException;
import com.github.thecybrix.simpleneuralnetwork.server.RequestHandler;
import com.github.thecybrix.simpleneuralnetwork.server.SessionContext;
import com.github.thecybrix.simpleneuralnetwork.util.intObjPair;

import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;

public class RawRequestDispatcher implements RequestHandler<RawRequest, RawResponse> {

    private ExecutorService threadPool = Executors.newWorkStealingPool();

    private Int2ObjectOpenHashMap<RequestHandler<RawRequest, RawResponse>> requestHandlers = new Int2ObjectOpenHashMap<>();
    private LinkedList<Consumer<Exception>> callbacks = new LinkedList<>();

    public RawRequestDispatcher(ExecutorService threadPool) {
        this.threadPool = threadPool;
    }

    public RawRequestDispatcher() {
        this(ForkJoinPool.commonPool());
    }


    public CompletableFuture<RawResponse> dispatch(SessionContext context, RawRequest request) throws NullPointerException {
        Objects.requireNonNull(context, "Context is null.");
        Objects.requireNonNull(request, "Request is null.");
        return CompletableFuture.supplyAsync(() -> selectAndRunHandler(context, request), threadPool);
    }

    private RawResponse selectAndRunHandler(SessionContext context, RawRequest request) throws RuntimeException {
        try {
            RequestHandler<RawRequest, RawResponse> handler = requestHandlers.getOrDefault(request.getEncodingType(), null);
            if (handler == null) throw new NoSuchRequestTypeException("Request type \"" + request.getEncodingType() + "\" is unbound.");
            return handler.handle(context, request);
        } catch (NoSuchRequestTypeException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public RawResponse handle(SessionContext context, RawRequest request) throws RuntimeException {
        try {
            return dispatch(context, request).get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }


    public RequestHandler<RawRequest, RawResponse> getRequestHandler(int type) {
        return requestHandlers.getOrDefault(type, null);
    }


    public void addRequestHandler(int type, RequestHandler<RawRequest, RawResponse> handler) throws EndpointConflictException {
        if (requestHandlers.containsKey(type)) {
            throw new EndpointConflictException("Type \"" + type +  "\" is already defined.");
        }
        requestHandlers.put(type, handler);
    }


    public void addRequestHandlers(Collection<intObjPair<RequestHandler<RawRequest, RawResponse>>> handlers) throws IllegalArgumentException, EndpointConflictException, NullPointerException {
        Objects.requireNonNull(handlers, "Hanlders is null.");
        ArrayList<Integer> typeNumbers = new ArrayList<>(handlers.size());
        if (handlers.stream().anyMatch(x -> x == null)) throw new NullPointerException("Handlers contains null.");
        if (handlers.stream().anyMatch(x -> x.getIntegerOrElse(0) < 0)) throw new IllegalArgumentException("Handler type cannot be negative.");
        handlers.stream().forEach(x -> typeNumbers.add(x.getIntegerOrElse(-1)));
        typeNumbers.sort(Comparator.naturalOrder());
        if (typeNumbers.stream().anyMatch(x -> x < 0)) throw new IllegalArgumentException("One or more handlers do not have a type numnber specified.");
        for (int i = 0; i < typeNumbers.size(); i++){
            if (typeNumbers.get(i) == typeNumbers.get(i + 1)){
                throw new EndpointConflictException("Two or more handlers have the same type number specified.");
            }
        }
        for (intObjPair<RequestHandler<RawRequest, RawResponse>> pair : handlers){
            requestHandlers.put(pair.getInteger(), pair.getObject());
        }
    }

    public RequestHandler<RawRequest, RawResponse> setRequestHandler(int type, RequestHandler<RawRequest, RawResponse> handler) {
        RequestHandler<RawRequest, RawResponse> prevHandler = requestHandlers.getOrDefault(type, null);
        requestHandlers.put(type, handler);
        return prevHandler;
    }

    /**
     * Removes the specified RequestHandler from this dispatcher.
     * @param handler
     * @return The type number associated with the handler, or {@code 0} (zero) if the dispatcher does not have this handler.
     */
    public int removeRequestHandler(RequestHandler<RawRequest, RawResponse> handler){
        for (Int2ObjectMap.Entry<RequestHandler<RawRequest, RawResponse>> entry : requestHandlers.int2ObjectEntrySet()) {
            if (Objects.equals(entry.getValue(), handler)) {
                int key = entry.getIntKey();
                requestHandlers.remove(key, handler);
                return key;
            }
        }
        return 0;
    }
    /**
     * Removes the RequestHandler for the specified type from this dispatcher.
     * @param type
     * @return The request handler for the given type, or {@code null} if the type is not defined.
     */
    public RequestHandler<RawRequest, RawResponse> removeRequestHandler(int type) {
        if (!requestHandlers.containsKey(type)) return null;
        return requestHandlers.remove(type);
    }

    @Override
    public List<Consumer<Exception>> getCallbackList() {
        return callbacks;
    }
}

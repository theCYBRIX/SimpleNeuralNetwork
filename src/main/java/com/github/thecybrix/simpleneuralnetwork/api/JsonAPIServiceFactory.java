package com.github.thecybrix.simpleneuralnetwork.api;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;

import com.github.thecybrix.simpleneuralnetwork.api.evolution.EvolutionContext;
import com.github.thecybrix.simpleneuralnetwork.api.idmanager.NetworkIDManager;
import com.github.thecybrix.simpleneuralnetwork.api.valuemapping.ValueMappingContext;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkBuilder;
import com.github.thecybrix.simpleneuralnetwork.server.EndpointsRequest;
import com.github.thecybrix.simpleneuralnetwork.server.JsonIOHandler;
import com.github.thecybrix.simpleneuralnetwork.server.JsonRequestHandler;
import com.github.thecybrix.simpleneuralnetwork.server.PropertyType;
import com.github.thecybrix.simpleneuralnetwork.server.ResponsePacket;
import com.github.thecybrix.simpleneuralnetwork.server.SimpleStdioServer;
import com.github.thecybrix.simpleneuralnetwork.server.SimpleTCPServer;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelector;
import com.google.gson.JsonObject;

public class JsonAPIServiceFactory {
    public static <E extends MutableNeuralNetwork> SimpleStdioServer createStdioServer(NeuralNetworkBuilder<E> networkBuilder, ParentSelector<E> parentSelector, Runnable onExit) {
        Objects.requireNonNull(networkBuilder, "Network builder is null.");
        parentSelector = (parentSelector != null) ? parentSelector : ParentSelector.eliteSelection();

        JsonIOHandler handler = createJsonHandler(networkBuilder, parentSelector);

        handler.addRequestHandler(new JsonRequestHandler() {
            @Override
            public String getEndpoint() {
                return "exit";
            }

            @Override
            public ResponsePacket handle(JsonObject request) throws Exception {
                if(onExit != null) onExit.run();
                handler.stop();
                return ResponsePacket.ok();
            }

            @Override
            public Map<String, PropertyType> getRequiredProperties() {
                return Collections.emptyMap();
            }

            @Override
            public Map<String, PropertyType> getOptionalProperties() {
                return Collections.emptyMap();
            }

            @Override
            public Map<String, PropertyType> getOutputProperties() {
                return Collections.emptyMap();
            }
        });

        return new SimpleStdioServer(handler);
    }

    public static <E extends MutableNeuralNetwork> SimpleStdioServer createStdioServer(NeuralNetworkBuilder<E> networkBuilder, ParentSelector<E> parentSelector) {
        return createStdioServer(networkBuilder, parentSelector, null);
    }

    public static <E extends MutableNeuralNetwork> SimpleStdioServer createStdioServer(NeuralNetworkBuilder<E> networkBuilder, Runnable onExit) {
        return createStdioServer(networkBuilder, null, onExit);
    }

    public static <E extends MutableNeuralNetwork> SimpleStdioServer createStdioServer(NeuralNetworkBuilder<E> networkBuilder) {
        return createStdioServer(networkBuilder, null, null);
    }

    public static <E extends MutableNeuralNetwork> SimpleTCPServer createTCPServer(int port, NeuralNetworkBuilder<E> networkBuilder, ParentSelector<E> parentSelector) throws IllegalArgumentException, NullPointerException {
        return new SimpleTCPServer(port, createJsonHandler(networkBuilder, parentSelector));
    }

    public static <E extends MutableNeuralNetwork> SimpleTCPServer createTCPServer(int port, NeuralNetworkBuilder<E> networkBuilder) throws IllegalArgumentException, NullPointerException {
        return createTCPServer(port, networkBuilder, null);
    }

    public static <E extends MutableNeuralNetwork> JsonIOHandler createJsonHandler(NeuralNetworkBuilder<E> networkBuilder, ParentSelector<E> parentSelector) {
        JsonIOHandler handler = new JsonIOHandler();

        NetworkIDManager<E> networkIdManager = new NetworkIDManager<>();
        EvolutionContext<E> evolutionContext = new EvolutionContext<>(networkIdManager, networkBuilder, parentSelector);
        ValueMappingContext<E> valueMappingContext = new ValueMappingContext<>(networkIdManager, networkBuilder, parentSelector);

        handler.addRequestHandlers(networkIdManager.getRequestHandlers());
        handler.addRequestHandlers(evolutionContext.getRequestHandlers());
        handler.addRequestHandlers(valueMappingContext.getRequestHandlers());

        handler.addRequestHandler(new EndpointsRequest(handler));

        return handler;
    }

    public static <E extends MutableNeuralNetwork> JsonIOHandler createJsonHandler(NeuralNetworkBuilder<E> networkBuilder) {
        return createJsonHandler(networkBuilder, null);
    }
}

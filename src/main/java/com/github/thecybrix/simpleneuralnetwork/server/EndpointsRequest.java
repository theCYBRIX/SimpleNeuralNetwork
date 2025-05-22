package com.github.thecybrix.simpleneuralnetwork.server;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;

import com.google.gson.JsonObject;

public class EndpointsRequest extends AbstractJsonRequestHandler{
    final private static String DEFAULT_ENDPOINT = "endpoints";
    final private static String ENDPOINT_PROPERTY = "endpoints";

    final private JsonIOHandler IO_HANDLER;

    enum IODirection {
        REQUEST,
        RESPONSE;
        
        @Override
        public String toString() {
            return name().toLowerCase();
        }
    }

    public EndpointsRequest(JsonIOHandler ioHandler) {
        this(ioHandler, DEFAULT_ENDPOINT);
    }

    public EndpointsRequest(JsonIOHandler ioHandler, String endpoint) {
        super(endpoint,
            NO_PROPERTIES,
            NO_PROPERTIES,
            Map.of(
                ENDPOINT_PROPERTY, PropertyType.mapOf(
                    PropertyType.STRING,
                    PropertyType.arrayOf(PropertyType.mapOf(PropertyType.STRING, PropertyType.OBJECT)))
            )
        );
        IO_HANDLER = Objects.requireNonNull(ioHandler, "APIIOHandler is null.");
    }

    @Override
    protected ResponsePacket handleRequest(JsonObject request) throws Exception {
        return ResponsePacket.message(getEndpointMap(IO_HANDLER));
    }

    public static Map<String, Object> getEndpointMap(JsonIOHandler ioHandler){
        return getEndpointMap(ioHandler.getRequestHandlers());
    }

    public static Map<String, Object> getEndpointMap(Collection<JsonRequestHandler> requestHandlers){
        ArrayList<Map<String, Object>> endpoints = new ArrayList<>(requestHandlers.size());
        for(JsonRequestHandler handler : requestHandlers)
            endpoints.add(getPropertiesMap(handler));
        return Collections.singletonMap("endpoints", endpoints);
    }

    public static Map<String, Object> getPropertiesMap(JsonRequestHandler handler){
        Map<String, Object> propertiesMap;
        
        Map<String, PropertyType> required = handler.getRequiredProperties();
        Map<String, PropertyType> optional = handler.getOptionalProperties();
        Map<String, PropertyType> response = handler.getOutputProperties();

        int num_handlers = required.size() + optional.size() + response.size();

        if(num_handlers > 0){
            ArrayList<Map<String, String>> properties = new ArrayList<>(num_handlers);
            
            for(Entry<String, PropertyType> prop : required.entrySet())
                properties.add(getRequestPropertyDetailsMap(prop.getKey(), prop.getValue(), false));
            for(Entry<String, PropertyType> prop : optional.entrySet())
                properties.add(getRequestPropertyDetailsMap(prop.getKey(), prop.getValue(), true));
            for(Entry<String, PropertyType> prop : response.entrySet())
                properties.add(getResponsePropertyDetailsMap(prop.getKey(), prop.getValue()));

            Map<String, List<Map<String, String>>> payload = Collections.singletonMap("properties", properties);

            propertiesMap = Map.of(
                JsonIOHandler.REQUEST_FIELDS[0], handler.getEndpoint(),
                JsonIOHandler.REQUEST_FIELDS[1], payload
            );
        
        } else {
            propertiesMap = Collections.singletonMap(JsonIOHandler.REQUEST_FIELDS[0], handler.getEndpoint());
        }
        
        return propertiesMap;
    }

    public static HashMap<String, String> getRequestPropertyDetailsMap(String name, PropertyType type, boolean optional){
        HashMap<String, String> map = getPropertyDetailsMap(name, type, IODirection.REQUEST);
        map.put("optional", Boolean.toString(optional));
        return map;
    }

    public static HashMap<String, String> getResponsePropertyDetailsMap(String name, PropertyType type){
        return getPropertyDetailsMap(name, type, IODirection.RESPONSE);
    }

    private static HashMap<String, String> getPropertyDetailsMap(String name, PropertyType type, IODirection direction){
        HashMap<String, String> map = new HashMap<>();
        map.put("name", name);
        map.put("type", type.toString());
        map.put("direction", direction.toString());
        return map;
    }
    
}

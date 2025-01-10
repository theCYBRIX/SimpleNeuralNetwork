package com.github.thecybrix.simpleneuralnetwork.api;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.function.Predicate;
import com.google.gson.JsonObject;

public abstract class AbstractRequestHandler implements RequestHandler {

    final protected static Map<String, PropertyType> NO_PROPERTIES = null;

    final private Map<String, PropertyType> REQUIRED_PROPERTIES;
    final private Map<String, PropertyType> OPTIONAL_PROPERTIES;

    final private String ENDPOINT;

    public AbstractRequestHandler(String endpoint, Map<String, PropertyType> requiredProperties, Map<String, PropertyType> optionalProperties) {
        REQUIRED_PROPERTIES = (requiredProperties != null) ? Collections.unmodifiableMap(new HashMap<>(requiredProperties)) : Collections.emptyMap();
        OPTIONAL_PROPERTIES = (requiredProperties != null) ? Collections.unmodifiableMap(new HashMap<>(optionalProperties)) : Collections.emptyMap();
        this.ENDPOINT = Objects.requireNonNull(endpoint, "Endpoint is null.");
    }

    protected abstract ResponsePacket handleRequest(JsonObject request) throws Exception;

    @Override
    final public String getKey() {
        return ENDPOINT;
    }
    
    @Override
    final public ResponsePacket handle(JsonObject request) throws Exception {
        Predicate<String> isMissingProperty = x -> !request.has(x);
        if(REQUIRED_PROPERTIES.keySet().stream().anyMatch(isMissingProperty)){
            String[] missing = REQUIRED_PROPERTIES.keySet().stream().filter(isMissingProperty).toArray(String[]::new);
            return missingProperty(missing);
        }

        return handleRequest(request);
    }

    protected ResponsePacket missingProperty(String... stringNames) throws IllegalArgumentException {
        if(stringNames == null) throw new IllegalArgumentException("String names array is null.");
        if(stringNames.length == 0) throw new IllegalArgumentException("String names array empty null.");

        HashMap<String, Map<String, String>> details = new HashMap<>();
        for (String property : stringNames) {
            if(REQUIRED_PROPERTIES.containsKey(property))
                details.put(property, getPropertyHelpMap(property, REQUIRED_PROPERTIES.get(property), false));
            else if(OPTIONAL_PROPERTIES.containsKey(property))
                details.put(property, getPropertyHelpMap(property, OPTIONAL_PROPERTIES.get(property), true));
            else
                throw new IllegalArgumentException("Property \"" + property + "\" not listed in either required or optional property list.");
        }

        return ResponsePacket.error("Missing Property", stringNames);
    }

    public String getHelpString(){
        ArrayList<Map<String, String>> properties = new ArrayList<>(REQUIRED_PROPERTIES.size() + OPTIONAL_PROPERTIES.size());
        for(Entry<String, PropertyType> prop : REQUIRED_PROPERTIES.entrySet())
            properties.add(getPropertyHelpMap(prop.getKey(), prop.getValue(), false));
        for(Entry<String, PropertyType> prop : OPTIONAL_PROPERTIES.entrySet())
            properties.add(getPropertyHelpMap(prop.getKey(), prop.getValue(), true));

        Map<String, List<Map<String, String>>> payload = new HashMap<>();
        payload.put("properties", properties);

        Map<String, Object> requestPacket = new HashMap<>();
        requestPacket.put(APIIOHandler.REQUEST_FIELDS[0], ENDPOINT);
        requestPacket.put(APIIOHandler.REQUEST_FIELDS[1], payload);

        return RequestHandlerUtils.GSON.toJson(requestPacket);
    }

    private Map<String, String> getPropertyHelpMap(String name, PropertyType type, boolean optional){
        Map<String, String> map = getPropertyHelpMap(name, optional);
        map.put("type", type.toString());
        return map;
    }

    private Map<String, String> getPropertyHelpMap(String name, boolean optional){
        HashMap<String, String> map = new HashMap<>();
        map.put("property", name);
        map.put("optional", Boolean.toString(optional));
        return map;
    }

    public final Map<String, PropertyType> getRequiredProperties(){
        return REQUIRED_PROPERTIES;
    }

    public final Map<String, PropertyType> getOptionalProperties(){
        return OPTIONAL_PROPERTIES;
    }
    
}

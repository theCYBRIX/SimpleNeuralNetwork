package com.github.thecybrix.simpleneuralnetwork.api;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.Predicate;
import com.google.gson.JsonObject;

public abstract class AbstractRequestHandler implements RequestHandler {

    final protected static Map<String, PropertyType> NO_PROPERTIES = null;

    final private Map<String, PropertyType> REQUIRED_PROPERTIES;
    final private Map<String, PropertyType> OPTIONAL_PROPERTIES;
    final private Map<String, PropertyType> OUTPUT_PROPERTIES;

    final private String ENDPOINT;

    public AbstractRequestHandler(String endpoint, Map<String, PropertyType> requiredProperties, Map<String, PropertyType> optionalProperties, Map<String, PropertyType> outputProperties) {
        ENDPOINT = Objects.requireNonNull(endpoint, "Endpoint is null.");
        REQUIRED_PROPERTIES = (requiredProperties != null) ? Collections.unmodifiableMap(new HashMap<>(requiredProperties)) : Collections.emptyMap();
        OPTIONAL_PROPERTIES = (optionalProperties != null) ? Collections.unmodifiableMap(new HashMap<>(optionalProperties)) : Collections.emptyMap();
        OUTPUT_PROPERTIES = (outputProperties != null) ? Collections.unmodifiableMap(new HashMap<>(outputProperties)) : Collections.emptyMap();
    }

    protected abstract ResponsePacket handleRequest(JsonObject request) throws Exception;

    @Override
    final public String getEndpoint() {
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

        ArrayList<Map<String, String>> details = new ArrayList<>(stringNames.length);
        for (String property : stringNames) {
            if(REQUIRED_PROPERTIES.containsKey(property))
                details.add(getPropertyHelpMap(property, REQUIRED_PROPERTIES.get(property), false));
            else if(OPTIONAL_PROPERTIES.containsKey(property))
                details.add(getPropertyHelpMap(property, OPTIONAL_PROPERTIES.get(property), true));
            else
                throw new IllegalArgumentException("Property \"" + property + "\" not listed in either required or optional property list.");
        }

        return ResponsePacket.error("Missing Property", details);
    }

    private Map<String, String> getPropertyHelpMap(String name, PropertyType type, boolean optional){
        Map<String, String> map = getPropertyHelpMap(name, optional);
        map.put("type", type.toString());
        return map;
    }

    private Map<String, String> getPropertyHelpMap(String name, boolean optional){
        HashMap<String, String> map = new HashMap<>();
        map.put("name", name);
        map.put("optional", Boolean.toString(optional));
        return map;
    }

    @Override
    public final Map<String, PropertyType> getRequiredProperties(){
        return REQUIRED_PROPERTIES;
    }

    @Override
    public final Map<String, PropertyType> getOptionalProperties(){
        return OPTIONAL_PROPERTIES;
    }

    @Override
    public final Map<String, PropertyType> getOutputProperties(){
        return OUTPUT_PROPERTIES;
    }
    
}

package com.github.thecybrix.simpleneuralnetwork.api;

import java.util.Map;

import com.google.gson.JsonObject;

public interface RequestHandler{
    public String getEndpoint();
    public ResponsePacket handle(JsonObject request) throws Exception;

    public Map<String, PropertyType> getRequiredProperties();
    public Map<String, PropertyType> getOptionalProperties();
    public Map<String, PropertyType> getOutputProperties();
}
package com.github.thecybrix.simpleneuralnetwork.server;

import java.util.Map;

import com.google.gson.JsonObject;

public interface JsonRequestHandler{
    public String getEndpoint();
    public ResponsePacket handle(JsonObject request) throws Exception;

    public Map<String, PropertyType> getRequiredProperties();
    public Map<String, PropertyType> getOptionalProperties();
    public Map<String, PropertyType> getOutputProperties();
}
package com.github.thecybrix.simpleneuralnetwork.api;

import com.google.gson.JsonObject;

public interface RequestHandler{
    public String getKey();
    public ResponsePacket handle(JsonObject request) throws Exception;
}
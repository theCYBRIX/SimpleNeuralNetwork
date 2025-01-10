package com.github.thecybrix.simpleneuralnetwork.api;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;


public class ResponsePacket {

    final public static String MESSAGE_KEY = "message",
                               DETAILS_KEY = "details";

    public enum Status {
        OK,
        ERROR;

        final private String READABLE_NAME;

        Status(){
            READABLE_NAME = this.name().toLowerCase();
        }

        Status(String readableName){
            READABLE_NAME = readableName;
        }

        @Override
        public String toString() {
            return READABLE_NAME;
        }
    }

    protected String status;
    protected Map<? extends Object, ? extends Object> payload;
    
    public ResponsePacket() {
        this(Status.OK);
    }
    
    public ResponsePacket(Status status) {
        this(status.toString());
    }
    
    public ResponsePacket(Status status, String message) {
        this(status.toString(), Collections.singletonMap(MESSAGE_KEY, message));
    }
    
    public ResponsePacket(String status) {
        this(status, null);
    }

    public ResponsePacket(Map<? extends Object, ? extends Object> payload) {
        this(Status.OK, payload);
    }

    public ResponsePacket(Status status, Map<? extends Object, ? extends Object> payload) {
        this(status.toString(), payload);
    }

    public ResponsePacket(String status, Map<? extends Object, ? extends Object> payload) {
        this.status = status;
        this.payload = payload;
    }

    public static ResponsePacket message(Map<? extends Object, ? extends Object> payload){
        return new ResponsePacket(payload);
    }

    public static ResponsePacket message(String fieldName, Map<? extends Object, ? extends Object> payload){
        return new ResponsePacket(Collections.singletonMap(fieldName, payload));
    }

    public static ResponsePacket message(String message){
        return new ResponsePacket(Status.OK, message);
    }

    public static ResponsePacket ok(){
        return new ResponsePacket();
    }
    
    public static ResponsePacket error(Exception e){
        return error(e.getMessage(), e);
    }
    
    public static ResponsePacket error(String message){
        return new ResponsePacket(Status.ERROR, message);
    }
    
    public static ResponsePacket error(String message, Exception e){
        return error(message, RequestHandlerUtils.stackTraceToString(e));
    }
    
    public static <T> ResponsePacket error(String message, T details){
        HashMap<Object, Object> data = new HashMap<>(2);
        data.put(MESSAGE_KEY, message);
        data.put(DETAILS_KEY, details);
        return new ResponsePacket(Status.ERROR, data);
    }
}
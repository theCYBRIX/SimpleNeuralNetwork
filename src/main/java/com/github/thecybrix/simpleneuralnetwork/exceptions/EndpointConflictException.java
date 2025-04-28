package com.github.thecybrix.simpleneuralnetwork.exceptions;

/**
    Thrown when attempting to add an already defined endpoint to the API.
*/
public class EndpointConflictException extends RuntimeException {
    public EndpointConflictException(){}
    public EndpointConflictException(String message){
        super(message);
    }
    public EndpointConflictException(Exception cause){
        super(cause);
    }
    public EndpointConflictException(String message, Exception cause){
        super(message, cause);
    }
}

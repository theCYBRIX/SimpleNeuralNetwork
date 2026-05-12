package com.github.thecybrix.simpleneuralnetwork.exceptions;

public class NoSuchRequestTypeException extends Exception{
    public NoSuchRequestTypeException(){}
    public NoSuchRequestTypeException(String message){
        super(message);
    }
    public NoSuchRequestTypeException(Exception cause){
        super(cause);
    }
    public NoSuchRequestTypeException(String message, Exception cause){
        super(message, cause);
    }
}

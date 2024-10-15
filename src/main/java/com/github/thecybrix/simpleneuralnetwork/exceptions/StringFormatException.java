package com.github.thecybrix.simpleneuralnetwork.exceptions;

public class StringFormatException extends IllegalArgumentException{
    public StringFormatException(){ super(); }
    public StringFormatException(String msg){ super(msg); }
}

package com.github.thecybrix.simpleneuralnetwork.exceptions;

public class LayoutMismatchException extends RuntimeException {
    /**
     * Indicates that two Neural Networks do not have the number of layers or nodes per layer, or the same activation functions for each layer; 
     */
    public LayoutMismatchException(){ super(); }
    /**
     * Indicates that two Neural Networks do not have the number of layers or nodes per layer, or the same activation functions for each layer; 
     */
    public LayoutMismatchException(String msg){ super(msg); }
}
package com.github.thecybrix.simpleneuralnetwork.core;

public interface NetworkBuilderFactory<E extends SimpleNeuralNetwork> {
    public NeuralNetworkBuilder<E> newBuilder();
}

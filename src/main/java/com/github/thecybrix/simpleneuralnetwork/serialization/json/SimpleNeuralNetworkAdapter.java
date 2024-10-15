package com.github.thecybrix.simpleneuralnetwork.serialization.json;

import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetworkBuilder;

public class SimpleNeuralNetworkAdapter extends NeuralNetworkAdapter<SimpleNeuralNetwork> {
    public SimpleNeuralNetworkAdapter(){ super(new SimpleNeuralNetworkBuilder()); }
    public SimpleNeuralNetworkAdapter(SimpleNeuralNetworkBuilder builder){ super(builder); }
}

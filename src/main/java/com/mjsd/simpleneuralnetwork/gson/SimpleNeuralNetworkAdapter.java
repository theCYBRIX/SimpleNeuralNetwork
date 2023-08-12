package com.mjsd.simpleneuralnetwork.gson;

import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetworkBuilder;

public class SimpleNeuralNetworkAdapter extends NeuralNetworkAdapter<SimpleNeuralNetwork> {
    public SimpleNeuralNetworkAdapter(){ super(new SimpleNeuralNetworkBuilder()); }
    public SimpleNeuralNetworkAdapter(SimpleNeuralNetworkBuilder builder){ super(builder); }
}

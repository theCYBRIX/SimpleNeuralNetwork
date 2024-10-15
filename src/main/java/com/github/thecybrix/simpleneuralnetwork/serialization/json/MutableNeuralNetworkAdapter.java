package com.github.thecybrix.simpleneuralnetwork.serialization.json;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetworkBuilder;

public class MutableNeuralNetworkAdapter extends NeuralNetworkAdapter<MutableNeuralNetwork> {
    public MutableNeuralNetworkAdapter(){ super(new MutableNeuralNetworkBuilder()); }
    public MutableNeuralNetworkAdapter(MutableNeuralNetworkBuilder builder){ super(builder); }
}
package com.mjsd.simpleneuralnetwork.gson;

import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetworkBuilder;

public class MutableNeuralNetworkAdapter extends NeuralNetworkAdapter<MutableNeuralNetwork> {
    public MutableNeuralNetworkAdapter(){ super(new MutableNeuralNetworkBuilder()); }
    public MutableNeuralNetworkAdapter(MutableNeuralNetworkBuilder builder){ super(builder); }
}
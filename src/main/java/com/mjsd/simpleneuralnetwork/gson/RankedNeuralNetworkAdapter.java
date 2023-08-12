package com.mjsd.simpleneuralnetwork.gson;

import com.mjsd.simpleneuralnetwork.training.RankedNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.RankedNeuralNetworkBuilder;

public class RankedNeuralNetworkAdapter extends NeuralNetworkAdapter<RankedNeuralNetwork> {
    public RankedNeuralNetworkAdapter(){ super(new RankedNeuralNetworkBuilder()); }
    public RankedNeuralNetworkAdapter(RankedNeuralNetworkBuilder builder){ super(builder); }
}

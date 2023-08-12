package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.function.Supplier;

import com.mjsd.simpleneuralnetwork.NeuralNetworkTools;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.evolution.SimpleOffspringProvider.TwoParentOffspringProvider;

public class SinglePointCrossover<E extends MutableNeuralNetwork> extends TwoParentOffspringProvider<E>{

    public SinglePointCrossover(Supplier<E> networkSupplier) throws NullPointerException {
        super((x, y) -> NeuralNetworkTools.singlePointCrossover(x, y, networkSupplier));
    }
    
}

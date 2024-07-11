package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.Objects;
import java.util.function.Supplier;

import com.mjsd.simpleneuralnetwork.NeuralNetworkTools;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.ScoredNetwork;
import com.mjsd.simpleneuralnetwork.training.evolution.SimpleOffspringGenerator.TwoParentOffspringProvider;

public class Crossover<E extends MutableNeuralNetwork, T extends Comparable<T>> extends TwoParentOffspringProvider<E, T>{

    public Crossover(Supplier<E> networkSupplier) throws NullPointerException {
        super((x, y) -> new ScoredNetwork<>(NeuralNetworkTools.crossover(x, y, networkSupplier.get())));
        Objects.requireNonNull(networkSupplier, "Network supplier is null.");
    }
    
}

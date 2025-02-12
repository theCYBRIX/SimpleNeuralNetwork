package com.github.thecybrix.simpleneuralnetwork.training.evolution.simple;

import java.util.Objects;
import java.util.function.Supplier;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkTools;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.simple.SimpleOffspringGenerator.TwoParentOffspringProvider;

public class Crossover<E extends MutableNeuralNetwork> extends TwoParentOffspringProvider<E>{

    public Crossover(Supplier<E> networkSupplier) throws NullPointerException {
        super((x, y, learningRate) -> new ScoredNetwork<>(NeuralNetworkTools.crossover(x, y, networkSupplier.get())), "crossover");
        Objects.requireNonNull(networkSupplier, "Network supplier is null.");
    }
    
}

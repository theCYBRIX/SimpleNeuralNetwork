package com.github.thecybrix.simpleneuralnetwork.training.evolution.simple;

import java.util.Objects;
import java.util.function.Supplier;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkTools;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.simple.SimpleOffspringGenerator.TwoParentOffspringProvider;

public class CrossoverPerNeuron<E extends MutableNeuralNetwork> extends TwoParentOffspringProvider<E>{

    public CrossoverPerNeuron(Supplier<E> networkSupplier) throws NullPointerException {
        super((x, y) -> new ScoredNetwork<>(NeuralNetworkTools.crossoverPerNeuron(x, y, networkSupplier.get())), "crossover (per neuron)");
        Objects.requireNonNull(networkSupplier, "Network supplier is null.");
    }
    
}

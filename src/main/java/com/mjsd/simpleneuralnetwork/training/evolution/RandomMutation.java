package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.Objects;
import java.util.function.Supplier;

import com.mjsd.simpleneuralnetwork.NeuralNetworkTools;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.ScoredNetwork;
import com.mjsd.simpleneuralnetwork.training.evolution.SimpleOffspringGenerator.SingleParentOffspringProvider;

public class RandomMutation<E extends MutableNeuralNetwork, T extends Comparable<T>> extends SingleParentOffspringProvider<E, T> {

    public RandomMutation(double maxWeightDeviation, double maxBiasDeviation, Supplier<E> networkSupplier) throws NullPointerException{
        super(x -> new ScoredNetwork<>(NeuralNetworkTools.mutate(NeuralNetworkTools.copyWeightsAndBiases(x, networkSupplier), maxWeightDeviation, maxBiasDeviation)));
        Objects.requireNonNull(networkSupplier, "Network supplier is null.");
    }

}

package com.github.thecybrix.simpleneuralnetwork.training.simple;

import java.util.Objects;
import java.util.function.Supplier;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkTools;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.simple.SimpleOffspringGenerator.SingleParentOffspringProvider;

public class RandomMutation<E extends MutableNeuralNetwork> extends SingleParentOffspringProvider<E> {

    public RandomMutation(double maxWeightDeviation, double maxBiasDeviation, Supplier<E> networkSupplier) throws NullPointerException{
        super(x -> new ScoredNetwork<>(NeuralNetworkTools.mutate(NeuralNetworkTools.copyWeightsAndBiases(x, networkSupplier), maxWeightDeviation, maxBiasDeviation)));
        Objects.requireNonNull(networkSupplier, "Network supplier is null.");
    }

}

package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.function.Supplier;

import com.mjsd.simpleneuralnetwork.NeuralNetworkTools;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.evolution.SimpleOffspringProvider.SingleParentOffspringProvider;

public class RandomMutation<E extends MutableNeuralNetwork> extends SingleParentOffspringProvider<E> {

    public RandomMutation(double maxWeightDeviation, double maxBiasDeviation, Supplier<E> networkSupplier) throws NullPointerException{
        super(x -> NeuralNetworkTools.randomMutation(NeuralNetworkTools.copy(x, networkSupplier), maxWeightDeviation, maxBiasDeviation));
    }

}

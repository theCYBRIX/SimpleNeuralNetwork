package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.Objects;
import java.util.function.Supplier;

import com.mjsd.simpleneuralnetwork.NeuralNetworkTools;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.evolution.SimpleOffspringProvider.TwoParentOffspringProvider;

public class LayerwiseCrossover<E extends MutableNeuralNetwork> extends TwoParentOffspringProvider<E> {
    public LayerwiseCrossover(Supplier<E> networkSupplier) throws NullPointerException {
        super((x, y) -> NeuralNetworkTools.layerwiseCrossover(x, y, networkSupplier));
        Objects.requireNonNull(networkSupplier, "Supplier is null.");
    }
}

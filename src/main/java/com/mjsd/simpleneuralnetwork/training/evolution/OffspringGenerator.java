package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.Collection;
import java.util.List;

import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.ScoredNetwork;

public interface OffspringGenerator<E extends MutableNeuralNetwork, T extends Comparable<T>> {
    public Collection<ScoredNetwork<E, T>> createOffspring(List<ScoredNetwork<E, T>> parents, int numOffspring);
}

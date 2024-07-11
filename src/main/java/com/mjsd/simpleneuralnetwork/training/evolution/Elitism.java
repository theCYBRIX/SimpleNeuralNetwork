package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.Collection;
import java.util.List;

import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.ScoredNetwork;

public class Elitism<E extends MutableNeuralNetwork, T extends Comparable<T>> implements OffspringGenerator<E, T> {
    @Override
    public Collection<ScoredNetwork<E, T>> createOffspring(List<ScoredNetwork<E, T>> parents, int numOffspring) {
        return parents.subList(Math.max(0, parents.size() - numOffspring), parents.size());
    }
}

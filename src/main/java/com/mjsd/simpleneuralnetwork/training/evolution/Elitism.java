package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.Collection;
import java.util.List;

import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;

public class Elitism<E extends MutableNeuralNetwork> implements OffspringProvider<E> {
    @Override
    public Collection<E> createOffspring(List<E> parents, int numOffspring) {
        return parents.subList(parents.size() - numOffspring, parents.size());
    }
}

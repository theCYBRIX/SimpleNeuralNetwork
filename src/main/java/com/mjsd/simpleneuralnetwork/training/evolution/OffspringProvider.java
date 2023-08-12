package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.Collection;
import java.util.List;

import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;

public interface OffspringProvider<E extends MutableNeuralNetwork> {
    public Collection<E> createOffspring(List<E> parents, int numOffspring);
}

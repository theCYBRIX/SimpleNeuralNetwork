package com.github.thecybrix.simpleneuralnetwork.training.evolution;

import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;

@FunctionalInterface
public interface OffspringFunction<E extends SimpleNeuralNetwork> {
    public ScoredNetwork<E> createOffspring(ScoredNetwork<E> parent, double learningRate);
}

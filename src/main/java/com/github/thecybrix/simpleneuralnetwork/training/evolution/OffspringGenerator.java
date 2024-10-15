package com.github.thecybrix.simpleneuralnetwork.training.evolution;

import java.util.List;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;

public interface OffspringGenerator<E extends MutableNeuralNetwork> {
    public List<ScoredNetwork<E>> createOffspring(List<ScoredNetwork<E>> parents, int numOffspring);
}

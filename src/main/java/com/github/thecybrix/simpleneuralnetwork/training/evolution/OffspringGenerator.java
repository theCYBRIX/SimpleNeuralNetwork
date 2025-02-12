package com.github.thecybrix.simpleneuralnetwork.training.evolution;

import java.util.List;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.util.Identifiable;

public interface OffspringGenerator<E extends MutableNeuralNetwork> extends Identifiable {
    public List<ScoredNetwork<E>> createOffspring(List<ScoredNetwork<E>> parents, int numOffspring, double learningRate);
}

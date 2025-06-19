package com.github.thecybrix.simpleneuralnetwork.training.evolution;
import java.util.List;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;

public interface ParentSelector<E extends MutableNeuralNetwork> {
    public List<ScoredNetwork<E>> getParents(int numParents, List<ScoredNetwork<E>> population) throws IllegalArgumentException;
    public String getType();
}

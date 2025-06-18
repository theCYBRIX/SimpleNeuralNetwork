package com.github.thecybrix.simpleneuralnetwork.training.evolution;

import java.util.ArrayList;
import java.util.List;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.simpleneuralnetwork.util.Identifiable;

public interface OffspringGenerator<E extends MutableNeuralNetwork> extends Identifiable {
    
    public void createOffspring(List<ScoredNetwork<E>> parents, int numOffspring, List<ScoredNetwork<E>> destination);

    default public List<ScoredNetwork<E>> createOffspring(List<ScoredNetwork<E>> parents, int numOffspring){
        ArrayList<ScoredNetwork<E>> destination = new ArrayList<>(numOffspring);
        createOffspring(parents, numOffspring, destination);
        return destination;
    }
}

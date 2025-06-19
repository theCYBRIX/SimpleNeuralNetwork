package com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelectors;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelector;

public class EliteSelector<E extends MutableNeuralNetwork> implements ParentSelector<E> {
    final public static String TYPE_NAME = "EliteSelector";
    final private Comparator<ScoredNetwork<E>> comparator;

    public EliteSelector(Comparator<ScoredNetwork<E>> comparator) {
        this.comparator = Objects.requireNonNull(comparator, "Comparator is null.");
    }

    @Override
    public List<ScoredNetwork<E>> getParents(int numParents, List<ScoredNetwork<E>> population) throws IllegalArgumentException {
        return getTopNetworks(numParents, population, comparator);
    }
    
    private static <E extends MutableNeuralNetwork> List<ScoredNetwork<E>> getTopNetworks(int numNetworks, List<ScoredNetwork<E>> population, Comparator<ScoredNetwork<E>> comparator) throws IllegalArgumentException{
        if(numNetworks > population.size()) throw new IllegalArgumentException("numNetworks is larger than population.size()");
        if(numNetworks < 0) throw new IllegalArgumentException("numNetworks is less than 0");
        if(numNetworks == 0) return Collections.emptyList();
        Collections.sort(population, comparator);
        return new ArrayList<>(population.subList(population.size() - numNetworks, population.size()));
    }

    @Override
    public String getType() {
        return TYPE_NAME;
    }
}

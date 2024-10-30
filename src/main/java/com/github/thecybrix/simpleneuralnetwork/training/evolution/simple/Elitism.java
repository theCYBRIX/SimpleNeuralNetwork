package com.github.thecybrix.simpleneuralnetwork.training.evolution.simple;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.OffspringGenerator;

public class Elitism<E extends MutableNeuralNetwork, T extends Comparable<T>> implements OffspringGenerator<E> {
    @Override
    public List<ScoredNetwork<E>> createOffspring(List<ScoredNetwork<E>> parents, int numOffspring) {
        if(numOffspring > parents.size()) throw new IllegalArgumentException("numOffspring is greater than parents.size()");

        ArrayList<ScoredNetwork<E>> elites = new ArrayList<>(numOffspring);
        addShallowCopies(parents.subList(parents.size() - numOffspring, parents.size()), elites);

        return elites;
    }

    private static <E extends MutableNeuralNetwork> void addShallowCopies(Collection<ScoredNetwork<E>> from, Collection<ScoredNetwork<E>> to){
        for(ScoredNetwork<E> network : from)
            to.add(shallowCopy(network));
    }

    private static <E extends MutableNeuralNetwork> ScoredNetwork<E> shallowCopy(ScoredNetwork<E> network){
        return new ScoredNetwork<E>(network.get(), network.getScore());
    }

    @Override
    public String getIdentifyer() {
        return "elitism";
    }
}

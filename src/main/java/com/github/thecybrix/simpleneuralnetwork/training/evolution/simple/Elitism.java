package com.github.thecybrix.simpleneuralnetwork.training.evolution.simple;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.OffspringGenerator;

public class Elitism<E extends MutableNeuralNetwork, T extends Comparable<T>> implements OffspringGenerator<E> {
    @Override
    public void createOffspring(List<ScoredNetwork<E>> parents, int numOffspring, List<ScoredNetwork<E>> destination) {
        Objects.requireNonNull(destination, "Destination list is null.");
        if(numOffspring > parents.size()) throw new IllegalArgumentException("numOffspring is greater than parents.size()");

        ArrayList<ScoredNetwork<E>> wrappedElites = new ArrayList<>(numOffspring);

        HashSet<E> elites = new HashSet<>(numOffspring);
        int index = parents.size() - 1;
        while(elites.size() < numOffspring){
            ScoredNetwork<E> p = parents.get(index);
            
            if(elites.add(p.get())){
                wrappedElites.add(p);
            }
            index -= 1;

            if(index < 0) break;
        }

        if(wrappedElites.size() < numOffspring){
            wrappedElites.addAll(parents.subList(parents.size() - (numOffspring - wrappedElites.size()), parents.size()));
        }

        addShallowCopies(wrappedElites, destination);
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

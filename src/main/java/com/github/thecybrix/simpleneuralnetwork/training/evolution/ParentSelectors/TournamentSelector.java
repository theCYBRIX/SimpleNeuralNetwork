package com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelectors;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.Random;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelector;
import com.github.thecybrix.simpleneuralnetwork.util.Fraction;

public class TournamentSelector<E extends MutableNeuralNetwork> implements ParentSelector<E>{
    final public static String TYPE_NAME = "Tournament";
    final private float tournamentFraction;
    final private Comparator<ScoredNetwork<E>> comparator;
    final private Random random = new Random();

    public TournamentSelector(Fraction tournamentSize) throws NullPointerException{
        this(tournamentSize, Comparator.naturalOrder());
    }

    public TournamentSelector(float tournamentFraction) throws IllegalArgumentException{
        this(tournamentFraction, Comparator.naturalOrder());
    }

    public TournamentSelector(Fraction tournamentSize, Comparator<ScoredNetwork<E>> comparator) throws NullPointerException{
        tournamentFraction = tournamentSize.floatValue();
        this.comparator = Objects.requireNonNull(comparator, "Comparator is null.");
    }

    public TournamentSelector(float tournamentFraction, Comparator<ScoredNetwork<E>> comparator) throws IllegalArgumentException{
        if(tournamentFraction < 0 || tournamentFraction > 1) throw new IllegalArgumentException("tournamentFraction is outside the range of [0, 1]");
        this.tournamentFraction = tournamentFraction;
        this.comparator = Objects.requireNonNull(comparator, "Comparator is null.");
    }

    @Override
    public List<ScoredNetwork<E>> getParents(int numParents, List<ScoredNetwork<E>> population) throws IllegalArgumentException {
        if(numParents > population.size()) throw new IllegalArgumentException("numParents is larger than population.size()");
        if(numParents == population.size()) return population;

        ArrayList<ScoredNetwork<E>> parents = new ArrayList<>(numParents);
        population = new LinkedList<>(population);
        int tournamentSize = Math.max(1, (int)(population.size() * tournamentFraction));

        for (int i = 0; i < numParents; i++) {
            Collections.shuffle(parents, random);
            List<ScoredNetwork<E>> participants = population.subList(0, Math.min(tournamentSize, population.size()));
            parents.add(tournament(participants));
        }

        return parents;
    }

    public ScoredNetwork<E> tournament(List<ScoredNetwork<E>> networks){
        networks.sort(comparator);
        return networks.remove(networks.size() - 1);
    }

    @Override
    public String getType() {
        return TYPE_NAME;
    }
}


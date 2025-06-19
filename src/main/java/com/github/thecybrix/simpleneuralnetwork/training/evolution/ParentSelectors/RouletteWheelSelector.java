package com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelectors;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.OptionalDouble;
import java.util.Random;
import java.util.stream.Collectors;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelector;

public class RouletteWheelSelector<E extends MutableNeuralNetwork> implements ParentSelector<E> {
    final public static String TYPE_NAME = "RouletteWheel";
    final private Random random = new Random();
    final private boolean preferLowestScore;

    public RouletteWheelSelector(boolean preferLowestScore) {
        this.preferLowestScore = preferLowestScore;
    }

    @Override
    public List<ScoredNetwork<E>> getParents(int numParents, List<ScoredNetwork<E>> population) throws IllegalArgumentException {
        return select(numParents, population, preferLowestScore, random);
    }

    public static <E extends MutableNeuralNetwork> List<ScoredNetwork<E>> select(int numItems, List<ScoredNetwork<E>> networks) {
        return select(numItems, networks, false);
    }

    public static <E extends MutableNeuralNetwork> List<ScoredNetwork<E>> select(int numItems, List<ScoredNetwork<E>> networks, boolean invert) {
        return select(numItems, networks, invert, new Random());
    }

    public static <E extends MutableNeuralNetwork> List<ScoredNetwork<E>> select(int numItems, List<ScoredNetwork<E>> networks, boolean invert, Random random) throws IllegalArgumentException{
        if(numItems <= 0) throw new IllegalArgumentException("Cannot select less than a single item. (numItems <= 0)");
        if(numItems > networks.size()) throw new IllegalArgumentException("Cannot select more items than there are in total.");

        Objects.requireNonNull(networks, "networks is null.");

        if(networks.parallelStream().anyMatch(x -> x == null)) throw new NullPointerException("A weight may not be null.");

        if(random == null) random = new Random();

        networks = networks.parallelStream()
                           .filter(x -> x.getScore().isPresent())
                           .collect(Collectors.toCollection(ArrayList::new));

        OptionalDouble smallestScore = networks.parallelStream()
                                        .mapToDouble(x -> x.getScore().getAsDouble())
                                        .min();

        //Offset all values to be in range [0, min + max]
        if(smallestScore.isPresent())
            networks.parallelStream()
                    .forEach(x -> x.setScore(x.getScore().getAsDouble() - smallestScore.getAsDouble()));

        if(invert){
            //Convert all values to 1/{value}
            networks.parallelStream()
                    .forEach(x -> x.setScore(1.0d / x.getScore().getAsDouble()));
        }

        Collections.sort(networks);

        Iterator<ScoredNetwork<E>> networkIterator = networks.iterator();
        double cumulativeWeight = networkIterator.next().getScore().getAsDouble();
        while(networkIterator.hasNext()){
            ScoredNetwork<E> objA = networkIterator.next();
            cumulativeWeight += objA.getScore().getAsDouble();
            objA.setScore(cumulativeWeight);
        }

        ArrayList<ScoredNetwork<E>> out = new ArrayList<>(numItems);
        final ScoredNetwork<E> SCORE_HOLDER = ScoredNetwork.empty();
        
        for(int i = 0; i < numItems; i++){
            SCORE_HOLDER.setScore(random.nextDouble() * cumulativeWeight);
            int selectedIndex = Collections.binarySearch(networks, SCORE_HOLDER);
            if(selectedIndex < 0) selectedIndex = -(selectedIndex + 1);

            ScoredNetwork<E> selected = networks.remove(selectedIndex);
            out.add(selected);


            double selectedScore = selected.getScore().getAsDouble();
            if(selectedIndex > 0){ 
                selectedScore -= networks.get(selectedIndex - 1).getScore().getAsDouble();
            }
            cumulativeWeight -= selectedScore;
            for (ScoredNetwork<E> n : networks.subList(selectedIndex, networks.size())) {
                n.setScore(n.getScore().getAsDouble() - selectedScore);
            }
            if(invert) selectedScore = 1.0d / selectedScore;
            if(smallestScore.isPresent()) selectedScore += smallestScore.getAsDouble();
            selected.setScore(selectedScore);
        }

        return out;
    }

    @Override
    public String getType() {
        return TYPE_NAME;
    }
}
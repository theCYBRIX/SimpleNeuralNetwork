package com.github.thecybrix.simpleneuralnetwork.training.evolution;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.OptionalDouble;
import java.util.Random;
import java.util.stream.Collectors;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.util.Fraction;

@FunctionalInterface
public interface ParentSelector<E extends MutableNeuralNetwork> {
    public List<ScoredNetwork<E>> getParents(int numParents, List<ScoredNetwork<E>> population) throws IllegalArgumentException;

    

    public static <E extends MutableNeuralNetwork> ParentSelector<E> rouletteWheelSelection(){
        return rouletteWheelSelection(false);
    }

    public static <E extends MutableNeuralNetwork> ParentSelector<E> rouletteWheelSelection(boolean preferLowestScore){
        return (numParents, population) -> RouletteWheelSelector.select(numParents, population, preferLowestScore);
    }

    public static <E extends MutableNeuralNetwork> ParentSelector<E> tournamentSelection(Fraction tournamentSize){
        return tournamentSelection(tournamentSize, Comparator.naturalOrder());
    }

    public static <E extends MutableNeuralNetwork> ParentSelector<E> tournamentSelection(float tournamentSize){
        return tournamentSelection(tournamentSize, Comparator.naturalOrder());
    }

    public static <E extends MutableNeuralNetwork> ParentSelector<E> tournamentSelection(Fraction tournamentSize, Comparator<ScoredNetwork<E>> comparator){
        return new TournamentSelection<>(tournamentSize, comparator);
    }

    public static <E extends MutableNeuralNetwork> ParentSelector<E> tournamentSelection(float tournamentSize, Comparator<ScoredNetwork<E>> comparator){
        return new TournamentSelection<>(tournamentSize, comparator);
    }

    public static <E extends MutableNeuralNetwork> ParentSelector<E> eliteSelection(){
        return eliteSelection(Comparator.naturalOrder());
    }

    public static <E extends MutableNeuralNetwork> ParentSelector<E> eliteSelection(Comparator<ScoredNetwork<E>> comparator){
        return (numParents, population) -> getTopNetworks(numParents, population, comparator);
    }

    private static <E extends MutableNeuralNetwork> List<ScoredNetwork<E>> getTopNetworks(int numNetworks, List<ScoredNetwork<E>> population, Comparator<ScoredNetwork<E>> comparator) throws IllegalArgumentException{
        if(numNetworks > population.size()) throw new IllegalArgumentException("numNetworks is larger than population.size()");
        if(numNetworks < 0) throw new IllegalArgumentException("numNetworks is less than 0");
        if(numNetworks == 0) return Collections.emptyList();
        int lastIndex = population.size() - 1;
        Collections.sort(population, comparator);
        return new ArrayList<>(population.subList(lastIndex - numNetworks, lastIndex));
    }

    public static class TournamentSelection<E extends MutableNeuralNetwork> implements ParentSelector<E>{
        private float tournamentFraction;
        private Comparator<ScoredNetwork<E>> comparator;

        public TournamentSelection(Fraction tournamentSize) throws NullPointerException{
            this(tournamentSize, Comparator.naturalOrder());
        }

        public TournamentSelection(float tournamentFraction) throws IllegalArgumentException{
            this(tournamentFraction, Comparator.naturalOrder());
        }

        public TournamentSelection(Fraction tournamentSize, Comparator<ScoredNetwork<E>> comparator) throws NullPointerException{
            tournamentFraction = tournamentSize.floatValue();
            this.comparator = Objects.requireNonNull(comparator, "Comparator is null.");
        }

        public TournamentSelection(float tournamentFraction, Comparator<ScoredNetwork<E>> comparator) throws IllegalArgumentException{
            if(tournamentFraction < 0 || tournamentFraction > 1) throw new IllegalArgumentException("tournamentFraction is outside the range of [0, 1]");
            this.tournamentFraction = tournamentFraction;
            this.comparator = Objects.requireNonNull(comparator, "Comparator is null.");
        }

        @Override
        public List<ScoredNetwork<E>> getParents(int numParents, List<ScoredNetwork<E>> population) throws IllegalArgumentException {
            if(numParents > population.size()) throw new IllegalArgumentException("numParents is larger than population.size()");
            if(numParents == population.size()) return population;

            Random random = new Random();
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

    }
    
    public static class RouletteWheelSelector {

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
    }
}

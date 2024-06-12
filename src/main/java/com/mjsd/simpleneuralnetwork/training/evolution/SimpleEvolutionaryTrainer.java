package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.Objects;
import java.util.function.Supplier;

import org.apache.commons.numbers.fraction.Fraction;

import com.mjsd.simpleneuralnetwork.training.RankedNeuralNetwork;

public class SimpleEvolutionaryTrainer<E extends RankedNeuralNetwork> extends EvolutionaryTrainer<E> {

	final private static Fraction DEF_PARENT_FRACTION = Fraction.of(5, 100);

	final private static byte ELITE_RATIO_TERM = 5;
	final private static byte SUBTLE_MUTATION_RATIO_TERM = 35;
	final private static byte MODERATE_MUTATION_TERM = 30;
	final private static byte AGGRESSIVE_MUTATION_RATIO_TERM = 30;

	final private static float HIGH_BIAS_ADJUST_RATE = 2f,
                               MEDIUM_BIAS_ADJUST_RATE = 0.08f,
                               LOW_BIAS_ADJUST_RATE = 0.02f,
                               HIGH_WEIGHT_ADJUST_RATE = 1f,
                               MEDIUM_WEIGHT_ADJUST_RATE = 0.01f,
                               LOW_WEIGHT_ADJUST_RATE = 0.002f;

    public SimpleEvolutionaryTrainer(int networksPerGeneration, Supplier<E> networkSupplier,
            TrainingScenario<E> trainingScenario) throws IllegalArgumentException, NullPointerException {
        super(createSingleSpecies(networksPerGeneration, networkSupplier, (x, y) -> x.compareTo(y)), trainingScenario);
    }

    public SimpleEvolutionaryTrainer(int networksPerGeneration, Supplier<E> networkSupplier, Comparator<E> comparator,
            TrainingScenario<E> trainingScenario) throws IllegalArgumentException, NullPointerException {
        super(createSingleSpecies(networksPerGeneration, networkSupplier, Objects.requireNonNull(comparator, "Comparator is null.")), trainingScenario);
    }

    public SimpleEvolutionaryTrainer(int networksPerGeneration, int numSpecies, Supplier<E> networkSupplier,
            TrainingScenario<E> trainingScenario) throws IllegalArgumentException, NullPointerException {
        super(getInstance(networksPerGeneration, numSpecies, networkSupplier, (x, y) -> x.compareTo(y)), trainingScenario);
    }

    public SimpleEvolutionaryTrainer(int networksPerGeneration, int numSpecies, Supplier<E> networkSupplier, Comparator<E> comparator,
            TrainingScenario<E> trainingScenario) throws IllegalArgumentException, NullPointerException {
        super(getInstance(networksPerGeneration, numSpecies, networkSupplier, Objects.requireNonNull(comparator, "Comparator is null.")), trainingScenario);
    }

    private static <E extends RankedNeuralNetwork> Population<E> getInstance(int totalNumNetworks, int numSpecies, Supplier<E> networkSupplier, Comparator<E> comparator){
        if(numSpecies < 1) throw new IllegalArgumentException("Illegal number of populations: " + numSpecies);
        return (numSpecies == 1) ? createSingleSpecies(totalNumNetworks, networkSupplier, comparator) : createEcosystem(totalNumNetworks, numSpecies, networkSupplier, comparator);
    }

    private static <E extends RankedNeuralNetwork> Ecosystem<E> createEcosystem(int totalNumNetworks, int numPopulations, Supplier<E> networkSupplier, Comparator<E> comparator) throws IllegalArgumentException, NullPointerException {
        return new Ecosystem<E>(totalNumNetworks, createSpecies(numPopulations, networkSupplier, comparator));
    }

    private static <E extends RankedNeuralNetwork> Species<E> createSingleSpecies(int totalNumNetworks, Supplier<E> networkSupplier, Comparator<E> comparator){
        return new Species<>(totalNumNetworks, DEF_PARENT_FRACTION, networkSupplier, getOffspringProviders(networkSupplier), getDistribution(), comparator);
    }

    private static <E extends RankedNeuralNetwork> Collection<Species<E>> createSpecies(int numSpecies, Supplier<E> networkSupplier, Comparator<E> comparator){
        ArrayList<Species<E>>  populations = new ArrayList<>(numSpecies);
        CompoundRatio distribution = getDistribution();
        Collection<OffspringProvider<E>> offspringProviders = getOffspringProviders(networkSupplier);

        for (int i = 0; i < numSpecies; i++)
            populations.add(new Species<>(DEF_PARENT_FRACTION, networkSupplier, offspringProviders, distribution, comparator));

        return populations;
    }

    private static <E extends RankedNeuralNetwork> Collection<OffspringProvider<E>> getOffspringProviders(Supplier<E> networkSupplier){
        return Arrays.asList(getEliteProvider(), getAggressiveMutationProvider(networkSupplier), getModerateMutationProvider(networkSupplier), getSubtleMutationProvider(networkSupplier));
    }

    private static <E extends RankedNeuralNetwork> OffspringProvider<E> getSubtleMutationProvider(Supplier<E> networkSupplier){
        return new RandomMutation<E>(LOW_WEIGHT_ADJUST_RATE, LOW_BIAS_ADJUST_RATE, networkSupplier);
    }

    private static <E extends RankedNeuralNetwork> OffspringProvider<E> getModerateMutationProvider(Supplier<E> networkSupplier){
        return new RandomMutation<E>(MEDIUM_WEIGHT_ADJUST_RATE, MEDIUM_BIAS_ADJUST_RATE, networkSupplier);
    }

    private static <E extends RankedNeuralNetwork> OffspringProvider<E> getAggressiveMutationProvider(Supplier<E> networkSupplier){
        return new RandomMutation<E>(HIGH_WEIGHT_ADJUST_RATE, HIGH_BIAS_ADJUST_RATE, networkSupplier);
    }

    /* 
    private static <E extends RankedNeuralNetwork> OffspringProvider<E> getCrossoverProvider(Supplier<E> networkSupplier){
        return new SinglePointCrossover<>(networkSupplier);
    }
    */
    private static <E extends RankedNeuralNetwork> OffspringProvider<E> getEliteProvider(){
        return new Elitism<>();
    }

    private static CompoundRatio getDistribution(){
        return CompoundRatio.of(ELITE_RATIO_TERM, MODERATE_MUTATION_TERM, AGGRESSIVE_MUTATION_RATIO_TERM, SUBTLE_MUTATION_RATIO_TERM);
    }

}

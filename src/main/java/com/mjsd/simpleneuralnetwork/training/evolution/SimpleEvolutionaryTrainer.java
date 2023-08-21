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

	final private static int ELITE_RATIO_TERM = 5;
	final private static int CROSSOVER_RATIO_TERM = 10;
	final private static int AGGRESSIVE_MUTATION_RATIO_TERM = 40;
	final private static int SUBTLE_MUTATION_RATIO_TERM = 45;

	final private static float HIGH_BIAS_ADJUST_RATE = 2f,
                               LOW_BIAS_ADJUST_RATE = 0.04f,
                               HIGH_WEIGHT_ADJUST_RATE = 1f,
                               LOW_WEIGHT_ADJUST_RATE = 0.002f;

    public SimpleEvolutionaryTrainer(int networksPerGeneration, Supplier<E> networkSupplier,
            TrainingScenario<E> trainingScenario) throws IllegalArgumentException, NullPointerException {
        super(getEcosystem(networksPerGeneration, networkSupplier, (x, y) -> x.compareTo(y)), trainingScenario);
    }

    public SimpleEvolutionaryTrainer(int networksPerGeneration, int numPopulations, Supplier<E> networkSupplier,
            TrainingScenario<E> trainingScenario) throws IllegalArgumentException, NullPointerException {
        super(getEcosystem(networksPerGeneration, numPopulations, networkSupplier, (x, y) -> x.compareTo(y)), trainingScenario);
    }

    public SimpleEvolutionaryTrainer(int networksPerGeneration, Supplier<E> networkSupplier, Comparator<E> comparator,
            TrainingScenario<E> trainingScenario) throws IllegalArgumentException, NullPointerException {
        super(getEcosystem(networksPerGeneration, networkSupplier, Objects.requireNonNull(comparator, "Comparator is null.")), trainingScenario);
    }

    public SimpleEvolutionaryTrainer(int networksPerGeneration, int numPopulations, Supplier<E> networkSupplier, Comparator<E> comparator,
            TrainingScenario<E> trainingScenario) throws IllegalArgumentException, NullPointerException {
        super(getEcosystem(networksPerGeneration, numPopulations, networkSupplier, Objects.requireNonNull(comparator, "Comparator is null.")), trainingScenario);
    }

    private static <E extends RankedNeuralNetwork> Ecosystem<E> getEcosystem(int totalNumNetworks, Supplier<E> networkSupplier, Comparator<E> comparator) throws IllegalArgumentException, NullPointerException {
        return new Ecosystem<>(totalNumNetworks, newPopulation(networkSupplier, comparator));
    }

    private static <E extends RankedNeuralNetwork> Ecosystem<E> getEcosystem(int totalNumNetworks, int numPopulations, Supplier<E> networkSupplier, Comparator<E> comparator) throws IllegalArgumentException, NullPointerException {
        if(numPopulations < 1) throw new  IllegalArgumentException("Illegal number of populations: " + numPopulations);
        return new Ecosystem<>(totalNumNetworks, newPopulations(numPopulations, networkSupplier, comparator));
    }

    private static <E extends RankedNeuralNetwork> Population<E> newPopulation(Supplier<E> networkSupplier, Comparator<E> comparator){
        return new Population<>(DEF_PARENT_FRACTION, networkSupplier, getOffspringProviders(networkSupplier), getDistribution(), comparator);
    }

    private static <E extends RankedNeuralNetwork> Collection<Population<E>> newPopulations(int numPopulations, Supplier<E> networkSupplier, Comparator<E> comparator){
        ArrayList<Population<E>>  populations = new ArrayList<>(numPopulations);
        CompoundRatio distribution = getDistribution();
        Collection<OffspringProvider<E>> offspringProviders = getOffspringProviders(networkSupplier);

        for (int i = 0; i < numPopulations; i++)
            populations.add(new Population<>(DEF_PARENT_FRACTION, networkSupplier, offspringProviders, distribution, comparator));

        return populations;
    }

    private static <E extends RankedNeuralNetwork> Collection<OffspringProvider<E>> getOffspringProviders(Supplier<E> networkSupplier){
        return Arrays.asList(getEliteProvider(), getCrossoverProvider(networkSupplier), getAggressiveMutationProvider(networkSupplier), getSubtleMutationProvider(networkSupplier));
    }

    private static <E extends RankedNeuralNetwork> OffspringProvider<E> getSubtleMutationProvider(Supplier<E> networkSupplier){
        return new RandomMutation<E>(LOW_WEIGHT_ADJUST_RATE, LOW_BIAS_ADJUST_RATE, networkSupplier);
    }

    private static <E extends RankedNeuralNetwork> OffspringProvider<E> getAggressiveMutationProvider(Supplier<E> networkSupplier){
        return new RandomMutation<E>(HIGH_WEIGHT_ADJUST_RATE, HIGH_BIAS_ADJUST_RATE, networkSupplier);
    }

    private static <E extends RankedNeuralNetwork> OffspringProvider<E> getCrossoverProvider(Supplier<E> networkSupplier){
        return new SinglePointCrossover<>(networkSupplier);
    }

    private static <E extends RankedNeuralNetwork> OffspringProvider<E> getEliteProvider(){
        return new Elitism<>();
    }

    private static CompoundRatio getDistribution(){
        return CompoundRatio.of(ELITE_RATIO_TERM, CROSSOVER_RATIO_TERM, AGGRESSIVE_MUTATION_RATIO_TERM, SUBTLE_MUTATION_RATIO_TERM);
    }

}

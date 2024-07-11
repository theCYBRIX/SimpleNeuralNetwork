package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.function.Supplier;

import org.apache.commons.numbers.fraction.Fraction;

import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.ScoredNetwork;

public class SimpleEvolutionaryTrainer<E extends MutableNeuralNetwork, T extends Comparable<T>> extends EvolutionaryTrainer<E, T> {

	final private static Fraction DEF_PARENT_FRACTION = Fraction.of(5, 100);

	final private static byte ELITE_RATIO_TERM = 5;
	final private static byte SUBTLE_MUTATION_RATIO_TERM = 35;
	final private static byte MODERATE_MUTATION_TERM = 30;
	final private static byte AGGRESSIVE_MUTATION_RATIO_TERM = 15;
	final private static byte CROSSOVER_RATIO_TERM = 15;

	final private static float HIGH_BIAS_ADJUST_RATE = 2f,
                               MEDIUM_BIAS_ADJUST_RATE = 0.08f,
                               LOW_BIAS_ADJUST_RATE = 0.02f,
                               HIGH_WEIGHT_ADJUST_RATE = 1f,
                               MEDIUM_WEIGHT_ADJUST_RATE = 0.01f,
                               LOW_WEIGHT_ADJUST_RATE = 0.002f;

    public SimpleEvolutionaryTrainer(int networksPerGeneration, Supplier<E> networkSupplier, TrainingScenario<E, T> trainingScenario) throws IllegalArgumentException, NullPointerException {
        super(networksPerGeneration, createEvolutionManager(networkSupplier), trainingScenario);
    }

    public SimpleEvolutionaryTrainer(int networksPerGeneration, Supplier<E> networkSupplier, TrainingScenario<E, T> trainingScenario, Comparator<ScoredNetwork<E, T>> comparator) throws IllegalArgumentException, NullPointerException {
        super(networksPerGeneration, createEvolutionManager(networkSupplier), trainingScenario, comparator);
    }

    private static <E extends MutableNeuralNetwork, T extends Comparable<T>> NetworkEvolutionManager<E, T> createEvolutionManager(Supplier<E> networkSupplier){
        return new NetworkEvolutionManager<E, T>(DEF_PARENT_FRACTION, networkSupplier, getOffspringProviders(networkSupplier), getDistribution());
    }

    private static <E extends MutableNeuralNetwork, T extends Comparable<T>> Collection<OffspringGenerator<E, T>> getOffspringProviders(Supplier<E> networkSupplier){
        return Arrays.asList(getEliteProvider(), getAggressiveMutationProvider(networkSupplier), getModerateMutationProvider(networkSupplier), getSubtleMutationProvider(networkSupplier), getCrossoverProvider(networkSupplier));
    }

    private static <E extends MutableNeuralNetwork, T extends Comparable<T>> OffspringGenerator<E, T> getSubtleMutationProvider(Supplier<E> networkSupplier){
        return new RandomMutation<E, T>(LOW_WEIGHT_ADJUST_RATE, LOW_BIAS_ADJUST_RATE, networkSupplier);
    }

    private static <E extends MutableNeuralNetwork, T extends Comparable<T>> OffspringGenerator<E, T> getModerateMutationProvider(Supplier<E> networkSupplier){
        return new RandomMutation<E, T>(MEDIUM_WEIGHT_ADJUST_RATE, MEDIUM_BIAS_ADJUST_RATE, networkSupplier);
    }

    private static <E extends MutableNeuralNetwork, T extends Comparable<T>> OffspringGenerator<E, T> getAggressiveMutationProvider(Supplier<E> networkSupplier){
        return new RandomMutation<E, T>(HIGH_WEIGHT_ADJUST_RATE, HIGH_BIAS_ADJUST_RATE, networkSupplier);
    }

    private static <E extends MutableNeuralNetwork, T extends Comparable<T>> OffspringGenerator<E, T> getCrossoverProvider(Supplier<E> networkSupplier){
        return new Crossover<E, T>(networkSupplier);
    }
        
    private static <E extends MutableNeuralNetwork, T extends Comparable<T>> OffspringGenerator<E, T> getEliteProvider(){
        return new Elitism<E, T>();
    }

    private static CompoundRatio getDistribution(){
        return CompoundRatio.of(ELITE_RATIO_TERM, MODERATE_MUTATION_TERM, AGGRESSIVE_MUTATION_RATIO_TERM, SUBTLE_MUTATION_RATIO_TERM, CROSSOVER_RATIO_TERM);
    }

}

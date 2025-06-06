package com.github.thecybrix.simpleneuralnetwork.training.evolution.simple;

import java.util.Arrays;
import java.util.Collection;
import java.util.function.Supplier;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.exceptions.DimensionsMismatchException;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.NetworkEvolutionManager;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.OffspringGenerator;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelector;
import com.github.thecybrix.simpleneuralnetwork.util.CompoundRatio;
import com.github.thecybrix.simpleneuralnetwork.util.Fraction;

public class SimpleEvolutionManager<E extends MutableNeuralNetwork> extends NetworkEvolutionManager<E> {

	final private static Fraction DEF_PARENT_FRACTION = Fraction.of(5, 100);

	final private static byte ELITE_RATIO_TERM = 3;
	final private static byte SUBTLE_MUTATION_RATIO_TERM = 30;
	final private static byte MODERATE_MUTATION_TERM = 30;
	final private static byte AGGRESSIVE_MUTATION_RATIO_TERM = 30;
	final private static byte CROSSOVER_RATIO_TERM = 7;

	final private static float HIGH_BIAS_ADJUST_RATE = 2f,
                               MEDIUM_BIAS_ADJUST_RATE = 0.08f,
                               LOW_BIAS_ADJUST_RATE = 0.02f,
                               HIGH_WEIGHT_ADJUST_RATE = 1f,
                               MEDIUM_WEIGHT_ADJUST_RATE = 0.01f,
                               LOW_WEIGHT_ADJUST_RATE = 0.002f;
    
    public SimpleEvolutionManager(Supplier<E> networkSupplier) throws DimensionsMismatchException, IllegalArgumentException, NullPointerException {
        this(networkSupplier, ParentSelector.eliteSelection());
    }
    
    public SimpleEvolutionManager(Supplier<E> networkSupplier, ParentSelector<E> parentSelector) throws DimensionsMismatchException, IllegalArgumentException, NullPointerException {
        super(DEF_PARENT_FRACTION, parentSelector, networkSupplier, getOffspringProviders(networkSupplier), getDistribution());
    }

    private static <E extends MutableNeuralNetwork> OffspringGenerator<E> getSubtleMutationProvider(Supplier<E> networkSupplier){
        return new RandomMutation<E>(LOW_WEIGHT_ADJUST_RATE, LOW_BIAS_ADJUST_RATE, networkSupplier, "subtle mutation", true);
    }

    private static <E extends MutableNeuralNetwork> OffspringGenerator<E> getModerateMutationProvider(Supplier<E> networkSupplier){
        return new RandomMutation<E>(MEDIUM_WEIGHT_ADJUST_RATE, MEDIUM_BIAS_ADJUST_RATE, networkSupplier, "moderate mutation", true);
    }

    private static <E extends MutableNeuralNetwork> OffspringGenerator<E> getAggressiveMutationProvider(Supplier<E> networkSupplier){
        return new RandomMutation<E>(HIGH_WEIGHT_ADJUST_RATE, HIGH_BIAS_ADJUST_RATE, networkSupplier, "aggressive mutation", true);
    }

    private static <E extends MutableNeuralNetwork> OffspringGenerator<E> getCrossoverProvider(Supplier<E> networkSupplier){
        return new Crossover<E>(networkSupplier);
    }
        
    private static <E extends MutableNeuralNetwork> OffspringGenerator<E> getEliteProvider(){
        return new Elitism<>();
    }

    private static <E extends MutableNeuralNetwork> Collection<OffspringGenerator<E>> getOffspringProviders(Supplier<E> networkSupplier){
        return Arrays.asList(getEliteProvider(), getAggressiveMutationProvider(networkSupplier), getModerateMutationProvider(networkSupplier), getSubtleMutationProvider(networkSupplier), getCrossoverProvider(networkSupplier));
    }

    private static CompoundRatio getDistribution(){
        return CompoundRatio.of(ELITE_RATIO_TERM, MODERATE_MUTATION_TERM, AGGRESSIVE_MUTATION_RATIO_TERM, SUBTLE_MUTATION_RATIO_TERM, CROSSOVER_RATIO_TERM);
    }
}

package com.github.thecybrix.simpleneuralnetwork.training.evolution.simple;

import java.util.Comparator;
import java.util.function.Supplier;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.EvolutionaryTrainer;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.NetworkEvolutionManager;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelector;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.TrainingScenario;

public class SimpleEvolutionaryTrainer<E extends MutableNeuralNetwork> extends EvolutionaryTrainer<E> {

    final private static byte DEFAULT_LEARNING_RATE = 2;

    public SimpleEvolutionaryTrainer(int networksPerGeneration, Supplier<E> networkSupplier, TrainingScenario<E> trainingScenario) throws IllegalArgumentException, NullPointerException {
        this(networksPerGeneration, networkSupplier, trainingScenario, null);
    }

    public SimpleEvolutionaryTrainer(int networksPerGeneration, Supplier<E> networkSupplier, TrainingScenario<E> trainingScenario, Comparator<ScoredNetwork<E>> comparator) throws IllegalArgumentException, NullPointerException {
        super(networksPerGeneration, createEvolutionManager(networkSupplier), trainingScenario, comparator);
        setDefaults();
    }

    public SimpleEvolutionaryTrainer(int networksPerGeneration, Supplier<E> networkSupplier, ParentSelector<E> parentSelector, TrainingScenario<E> trainingScenario) throws IllegalArgumentException, NullPointerException {
        this(networksPerGeneration, networkSupplier, parentSelector, trainingScenario, null);
    }

    public SimpleEvolutionaryTrainer(int networksPerGeneration, Supplier<E> networkSupplier, ParentSelector<E> parentSelector, TrainingScenario<E> trainingScenario, Comparator<ScoredNetwork<E>> comparator) throws IllegalArgumentException, NullPointerException {
        super(networksPerGeneration, createEvolutionManager(networkSupplier, parentSelector), trainingScenario, comparator);
        setDefaults();
    }

    private void setDefaults(){
        setLearningRate(DEFAULT_LEARNING_RATE);
        attachCallback(new LearningRateDecay<>());
    }

    private static <E extends MutableNeuralNetwork> NetworkEvolutionManager<E> createEvolutionManager(Supplier<E> networkSupplier){
        return createEvolutionManager(networkSupplier, ParentSelector.eliteSelection());
    }

    private static <E extends MutableNeuralNetwork> NetworkEvolutionManager<E> createEvolutionManager(Supplier<E> networkSupplier, ParentSelector<E> parentSelector){
        return new SimpleEvolutionManager<E>(networkSupplier, parentSelector);
    }

}

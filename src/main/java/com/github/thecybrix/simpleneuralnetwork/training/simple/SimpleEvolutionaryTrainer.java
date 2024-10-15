package com.github.thecybrix.simpleneuralnetwork.training.simple;

import java.util.Comparator;
import java.util.function.Supplier;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.EvolutionaryTrainer;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.NetworkEvolutionManager;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelector;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.TrainingScenario;

public class SimpleEvolutionaryTrainer<E extends MutableNeuralNetwork> extends EvolutionaryTrainer<E> {

    public SimpleEvolutionaryTrainer(int networksPerGeneration, Supplier<E> networkSupplier, TrainingScenario<E> trainingScenario) throws IllegalArgumentException, NullPointerException {
        super(networksPerGeneration, createEvolutionManager(networkSupplier), trainingScenario);
    }

    public SimpleEvolutionaryTrainer(int networksPerGeneration, Supplier<E> networkSupplier, TrainingScenario<E> trainingScenario, Comparator<ScoredNetwork<E>> comparator) throws IllegalArgumentException, NullPointerException {
        super(networksPerGeneration, createEvolutionManager(networkSupplier), trainingScenario, comparator);
    }

    public SimpleEvolutionaryTrainer(int networksPerGeneration, Supplier<E> networkSupplier, ParentSelector<E> parentSelector, TrainingScenario<E> trainingScenario) throws IllegalArgumentException, NullPointerException {
        super(networksPerGeneration, createEvolutionManager(networkSupplier, parentSelector), trainingScenario);
    }

    public SimpleEvolutionaryTrainer(int networksPerGeneration, Supplier<E> networkSupplier, ParentSelector<E> parentSelector, TrainingScenario<E> trainingScenario, Comparator<ScoredNetwork<E>> comparator) throws IllegalArgumentException, NullPointerException {
        super(networksPerGeneration, createEvolutionManager(networkSupplier, parentSelector), trainingScenario, comparator);
    }

    private static <E extends MutableNeuralNetwork> NetworkEvolutionManager<E> createEvolutionManager(Supplier<E> networkSupplier){
        return createEvolutionManager(networkSupplier, ParentSelector.eliteSelection());
    }

    private static <E extends MutableNeuralNetwork> NetworkEvolutionManager<E> createEvolutionManager(Supplier<E> networkSupplier, ParentSelector<E> parentSelector){
        return new SimpleEvolutionManager<E>(networkSupplier, parentSelector);
    }

}

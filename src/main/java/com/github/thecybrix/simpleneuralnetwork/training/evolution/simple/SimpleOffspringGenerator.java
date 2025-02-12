package com.github.thecybrix.simpleneuralnetwork.training.evolution.simple;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.BiParentOffspringFunction;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.OffspringFunction;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.OffspringGenerator;

public abstract class SimpleOffspringGenerator<E extends MutableNeuralNetwork> implements OffspringGenerator<E> {
    final public static String NO_IDENTIFYER = "unspecified";

    final private String IDENTIFYER;

    protected SimpleOffspringGenerator(String identifyer){
        IDENTIFYER = Objects.requireNonNull(identifyer, "Identifyer may not be null.");
    }

    final public static <E extends MutableNeuralNetwork> OffspringGenerator<E> createInstance(OffspringFunction<E> offspringFunction) throws NullPointerException{
        return createInstance(offspringFunction, NO_IDENTIFYER);
    }

    final public static <E extends MutableNeuralNetwork> OffspringGenerator<E> createInstance(OffspringFunction<E> offspringFunction, String name) throws NullPointerException{
        return new SingleParentOffspringProvider<>(offspringFunction, name);
    }

    final public static <E extends MutableNeuralNetwork> OffspringGenerator<E> createInstance(BiParentOffspringFunction<E> offspringFunction) throws NullPointerException{
        return createInstance(offspringFunction, NO_IDENTIFYER);
    }

    final public static <E extends MutableNeuralNetwork> OffspringGenerator<E> createInstance(BiParentOffspringFunction<E> offspringFunction, String name) throws NullPointerException{
        return new TwoParentOffspringProvider<>(offspringFunction, name);
    }

    protected abstract ScoredNetwork<E> applyAtIndex(List<ScoredNetwork<E>> parents, int index, double learningRate);

    @Override
    final public List<ScoredNetwork<E>> createOffspring(List<ScoredNetwork<E>> parents, int numOffspring, double learningRate) {
        if(parents.isEmpty() || numOffspring <= 0) return Collections.emptyList();
        ArrayList<ScoredNetwork<E>> offspring = new ArrayList<>(numOffspring);

        for(int i = 0; i < numOffspring; i++)
            offspring.add(applyAtIndex(parents, parents.size() - (1 + i%parents.size()), learningRate));

        return offspring;
    }

    @Override
    public String getIdentifyer() {
        return IDENTIFYER;
    }

    protected static class TwoParentOffspringProvider<E extends MutableNeuralNetwork> extends SimpleOffspringGenerator<E>{
        final protected BiParentOffspringFunction<E> OFFSPRING_FUNCTION;

        public TwoParentOffspringProvider(BiParentOffspringFunction<E> offspringFunction, String identifyer) throws NullPointerException{
            super(identifyer);
            OFFSPRING_FUNCTION = Objects.requireNonNull(offspringFunction);
        }

        @Override
        protected ScoredNetwork<E> applyAtIndex(List<ScoredNetwork<E>> parents, int index, double learningRate) {
            int nextIndex = index + 1;
            if(nextIndex == parents.size()) nextIndex = 0;

            return OFFSPRING_FUNCTION.createOffspring(parents.get(index), parents.get(nextIndex), learningRate);
        }

    }

    protected static class SingleParentOffspringProvider<E extends MutableNeuralNetwork> extends SimpleOffspringGenerator<E>{
        final protected OffspringFunction<E> OFFSPRING_FUNCTION;

        public SingleParentOffspringProvider(OffspringFunction<E> offspringFunction, String identifyer) throws NullPointerException{
            super(identifyer);
            OFFSPRING_FUNCTION = Objects.requireNonNull(offspringFunction);
        }

        @Override
        protected ScoredNetwork<E> applyAtIndex(List<ScoredNetwork<E>> parents, int index, double learningRate) {
            return OFFSPRING_FUNCTION.createOffspring(parents.get(index), learningRate);
        }

    }
    
}

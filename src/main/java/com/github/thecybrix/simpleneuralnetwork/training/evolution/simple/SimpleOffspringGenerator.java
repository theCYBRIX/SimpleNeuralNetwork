package com.github.thecybrix.simpleneuralnetwork.training.evolution.simple;

import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.OffspringGenerator;

public abstract class SimpleOffspringGenerator<E extends MutableNeuralNetwork> implements OffspringGenerator<E> {
    final public static String NO_IDENTIFYER = "unspecified";

    final private String IDENTIFYER;

    protected SimpleOffspringGenerator(String identifyer){
        IDENTIFYER = Objects.requireNonNull(identifyer, "Identifyer may not be null.");
    }

    final public static <E extends MutableNeuralNetwork> OffspringGenerator<E> createInstance(Function<ScoredNetwork<E>, ScoredNetwork<E>> offspringFunction) throws NullPointerException{
        return createInstance(offspringFunction, NO_IDENTIFYER);
    }

    final public static <E extends MutableNeuralNetwork> OffspringGenerator<E> createInstance(Function<ScoredNetwork<E>, ScoredNetwork<E>> offspringFunction, String name) throws NullPointerException{
        return new SingleParentOffspringProvider<>(offspringFunction, name);
    }

    final public static <E extends MutableNeuralNetwork> OffspringGenerator<E> createInstance(BiFunction<ScoredNetwork<E>, ScoredNetwork<E>, ScoredNetwork<E>> offspringFunction) throws NullPointerException{
        return createInstance(offspringFunction, NO_IDENTIFYER);
    }

    final public static <E extends MutableNeuralNetwork> OffspringGenerator<E> createInstance(BiFunction<ScoredNetwork<E>, ScoredNetwork<E>, ScoredNetwork<E>> offspringFunction, String name) throws NullPointerException{
        return new TwoParentOffspringProvider<>(offspringFunction, name);
    }

    protected abstract ScoredNetwork<E> applyAtIndex(List<ScoredNetwork<E>> parents, int index);

    @Override
    final public void createOffspring(List<ScoredNetwork<E>> parents, int numOffspring, List<ScoredNetwork<E>> destination) {
        Objects.requireNonNull(destination, "Destination list is null.");
        if(parents.isEmpty() || numOffspring <= 0) return;

        for(int i = 0; i < numOffspring; i++)
            destination.add(applyAtIndex(parents, parents.size() - (1 + i%parents.size())));
    }

    @Override
    public String getIdentifyer() {
        return IDENTIFYER;
    }

    protected static class TwoParentOffspringProvider<E extends MutableNeuralNetwork> extends SimpleOffspringGenerator<E>{
        final protected BiFunction<ScoredNetwork<E>, ScoredNetwork<E>, ScoredNetwork<E>> OFFSPRING_FUNCTION;

        public TwoParentOffspringProvider(BiFunction<ScoredNetwork<E>, ScoredNetwork<E>, ScoredNetwork<E>> offspringFunction, String identifyer) throws NullPointerException{
            super(identifyer);
            OFFSPRING_FUNCTION = Objects.requireNonNull(offspringFunction);
        }

        @Override
        protected ScoredNetwork<E> applyAtIndex(List<ScoredNetwork<E>> parents, int index) {
            int nextIndex = index + 1;
            if(nextIndex == parents.size()) nextIndex = 0;

            return OFFSPRING_FUNCTION.apply(parents.get(index), parents.get(nextIndex));
        }

    }

    protected static class SingleParentOffspringProvider<E extends MutableNeuralNetwork> extends SimpleOffspringGenerator<E>{
        final protected Function<ScoredNetwork<E>, ScoredNetwork<E>> OFFSPRING_FUNCTION;

        public SingleParentOffspringProvider(Function<ScoredNetwork<E>, ScoredNetwork<E>> offspringFunction, String identifyer) throws NullPointerException{
            super(identifyer);
            OFFSPRING_FUNCTION = Objects.requireNonNull(offspringFunction);
        }

        @Override
        protected ScoredNetwork<E> applyAtIndex(List<ScoredNetwork<E>> parents, int index) {
            return OFFSPRING_FUNCTION.apply(parents.get(index));
        }

    }
    
}

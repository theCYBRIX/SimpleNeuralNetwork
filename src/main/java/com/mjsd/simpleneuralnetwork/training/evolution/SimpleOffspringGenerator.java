package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;

import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.ScoredNetwork;

public abstract class SimpleOffspringGenerator<E extends MutableNeuralNetwork, T extends Comparable<T>> implements OffspringGenerator<E, T> {

    final public static <E extends MutableNeuralNetwork, T extends Comparable<T>> OffspringGenerator<E, T> createInstance(Function<ScoredNetwork<E, T>, ScoredNetwork<E, T>> offspringFunction) throws NullPointerException{
        return new SingleParentOffspringProvider<>(offspringFunction);
    }

    final public static <E extends MutableNeuralNetwork, T extends Comparable<T>> OffspringGenerator<E, T> createInstance(BiFunction<ScoredNetwork<E, T>, ScoredNetwork<E, T>, ScoredNetwork<E, T>> offspringFunction) throws NullPointerException{
        return new TwoParentOffspringProvider<>(offspringFunction);
    }

    @Override
    final public Collection<ScoredNetwork<E, T>> createOffspring(List<ScoredNetwork<E, T>> parents, int numOffspring) {
        if(parents.isEmpty() || numOffspring <= 0) return Collections.emptyList();
        ArrayList<ScoredNetwork<E, T>> offspring = new ArrayList<>(numOffspring);

        for(int i = numOffspring; i > 0; i--)
            offspring.add(applyAtIndex(parents, i%parents.size()));

        return offspring;
    }

    protected abstract ScoredNetwork<E, T> applyAtIndex(List<ScoredNetwork<E, T>> parents, int index);

    protected static class TwoParentOffspringProvider<E extends MutableNeuralNetwork, T extends Comparable<T>> extends SimpleOffspringGenerator<E, T>{
        final protected BiFunction<ScoredNetwork<E, T>, ScoredNetwork<E, T>, ScoredNetwork<E, T>> OFFSPRING_FUNCTION;

        public TwoParentOffspringProvider(BiFunction<ScoredNetwork<E, T>, ScoredNetwork<E, T>, ScoredNetwork<E, T>> offspringFunction) throws NullPointerException{
            OFFSPRING_FUNCTION = Objects.requireNonNull(offspringFunction);
        }

        @Override
        protected ScoredNetwork<E, T> applyAtIndex(List<ScoredNetwork<E, T>> parents, int index) {
            int nextIndex = index + 1;
            if(nextIndex == parents.size()) nextIndex = 0;

            return OFFSPRING_FUNCTION.apply(parents.get(index), parents.get(nextIndex));
        }

    }

    protected static class SingleParentOffspringProvider<E extends MutableNeuralNetwork, T extends Comparable<T>> extends SimpleOffspringGenerator<E, T>{
        final protected Function<ScoredNetwork<E, T>, ScoredNetwork<E, T>> OFFSPRING_FUNCTION;

        public SingleParentOffspringProvider(Function<ScoredNetwork<E, T>, ScoredNetwork<E, T>> offspringFunction) throws NullPointerException{
            OFFSPRING_FUNCTION = Objects.requireNonNull(offspringFunction);
        }

        @Override
        protected ScoredNetwork<E, T> applyAtIndex(List<ScoredNetwork<E, T>> parents, int index) {
            return OFFSPRING_FUNCTION.apply(parents.get(index));
        }

    }
    
}

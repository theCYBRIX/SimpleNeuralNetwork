package mjsd.simpleneuralnetwork.training.evolution;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;

import mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;

public abstract class SimpleOffspringProvider<E extends MutableNeuralNetwork> implements OffspringProvider<E> {

    final public static <E extends MutableNeuralNetwork> OffspringProvider<E> createInstance(Function<E, E> offspringFunction) throws NullPointerException{
        return new SingleParentOffspringProvider<>(offspringFunction);
    }

    final public static <E extends MutableNeuralNetwork> OffspringProvider<E> createInstance(BiFunction<E, E, E> offspringFunction) throws NullPointerException{
        return new TwoParentOffspringProvider<>(offspringFunction);
    }

    @Override
    final public Collection<E> createOffspring(List<E> parents, int numOffspring) {
        ArrayList<E> offspring = new ArrayList<>(numOffspring);

        for(int i = numOffspring; i > 0; i--)
            offspring.add(applyAtIndex(parents, i%parents.size()));

        return offspring;
    }

    protected abstract E applyAtIndex(List<E> parents, int index);

    protected static class TwoParentOffspringProvider<E extends MutableNeuralNetwork> extends SimpleOffspringProvider<E>{
        final protected BiFunction<E, E, E> OFFSPRING_FUNCTION;

        public TwoParentOffspringProvider(BiFunction<E, E, E> offspringFunction) throws NullPointerException{
            OFFSPRING_FUNCTION = Objects.requireNonNull(offspringFunction);
        }

        @Override
        protected E applyAtIndex(List<E> parents, int index) {
            int nextIndex = index + 1;
            if(nextIndex == parents.size()) nextIndex = 0;

            return OFFSPRING_FUNCTION.apply(parents.get(index), parents.get(nextIndex));
        }

    }

    protected static class SingleParentOffspringProvider<E extends MutableNeuralNetwork> extends SimpleOffspringProvider<E>{
        final protected Function<E, E> OFFSPRING_FUNCTION;

        public SingleParentOffspringProvider(Function<E, E> offspringFunction) throws NullPointerException{
            OFFSPRING_FUNCTION = Objects.requireNonNull(offspringFunction);
        }

        @Override
        protected E applyAtIndex(List<E> parents, int index) {
            return OFFSPRING_FUNCTION.apply(parents.get(index));
        }

    }
    
}

package com.github.thecybrix.simpleneuralnetwork.training.evolution;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.util.Fraction;
import com.github.thecybrix.simpleneuralnetwork.util.MultiPartRatio;
import com.github.thecybrix.simpleneuralnetwork.util.ObjIntPair;

public class EvolutionManagerBuilder<E extends MutableNeuralNetwork> {
    private Fraction parentFraction;
    private ParentSelector<E> parentSelector;
    private Supplier<E> networkSupplier;
    private ArrayList<ObjIntPair<OffspringGenerator<E>>> offspringProviders = new ArrayList<>();

    public EvolutionManagerBuilder(EvolutionManagerBuilder<E> template){
        parentFraction = template.parentFraction;
        parentSelector = template.parentSelector;
        networkSupplier = template.networkSupplier;
        offspringProviders = offspringProviders.stream().map(x -> x.copy()).collect(Collectors.toCollection(ArrayList::new));
    }

    public EvolutionManagerBuilder<E> reset(){
        parentFraction = null;
        parentSelector = null;
        networkSupplier = null;
        offspringProviders.clear();
        return this;
    }

    public EvolutionManagerBuilder<E> withParentFraction(Fraction fraction){
        parentFraction = Objects.requireNonNull(fraction, "Parent fraction is null.");
        return this;
    }

    public EvolutionManagerBuilder<E> withParentFraction(int numerator, int denominator) throws IllegalArgumentException {
        parentFraction = Fraction.of(numerator, denominator);
        return this;
    }

    public EvolutionManagerBuilder<E> withParentSelector(ParentSelector<E> selector) throws IllegalArgumentException {
        parentSelector = Objects.requireNonNull(selector, "Parent selector is null.");
        return this;
    }

    public EvolutionManagerBuilder<E> withNetworkSupplier(Supplier<E> supplier) {
        networkSupplier = supplier;
        return this;
    }

    public EvolutionManagerBuilder<E> addOffspringGenerator(OffspringGenerator<E> generator){
        return addOffspringGenerator(generator, 1);
    }

    public EvolutionManagerBuilder<E> addOffspringGenerator(OffspringGenerator<E> generator, int offspringFractionTerm){
        offspringProviders.add(ObjIntPair.of(generator, offspringFractionTerm));
        return this;
    }

    public NetworkEvolutionManager<E> build() throws IllegalStateException {
        if(parentFraction == null) throw new IllegalStateException("Parent fraction has not been set.");
        if(parentSelector == null) throw new IllegalStateException("Parent selector has not been set.");
        if(networkSupplier == null) throw new IllegalStateException("Network supplier has not been set.");
        if(offspringProviders.isEmpty()) throw new IllegalStateException("No offspring providers have been set.");
        List<OffspringGenerator<E>> offspringGenerators = offspringProviders.stream().map(x -> x.getObject()).collect(Collectors.toList());
        MultiPartRatio offspringRatio = MultiPartRatio.of(offspringProviders.stream().mapToInt(x -> x.getInteger()).toArray());
        return new NetworkEvolutionManager<>(parentFraction, parentSelector, networkSupplier, offspringGenerators, offspringRatio);
    }

}

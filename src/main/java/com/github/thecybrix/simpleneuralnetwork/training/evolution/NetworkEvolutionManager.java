package com.github.thecybrix.simpleneuralnetwork.training.evolution;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.Supplier;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkTools;
import com.github.thecybrix.simpleneuralnetwork.exceptions.DimensionsMismatchException;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.simpleneuralnetwork.util.MultiPartRatio;
import com.github.thecybrix.simpleneuralnetwork.util.ObjIntPair;
import com.github.thecybrix.simpleneuralnetwork.util.Fraction;

public class NetworkEvolutionManager<E extends MutableNeuralNetwork>{

    final private MultiPartRatio networkDistribution;
    final private List<OffspringGenerator<E>> offspringGenerators;

    private boolean parallel = false, createMetadata = false;

    private int[] networksPerProvider;
    private int expectedOffspring;

    private NewGenerationFunction<E> newGenerationFunction = this::linearNewGeneration;
    private ParentSelector<E> parentSelector;

    private float parentFraction;
    private Supplier<E> networkSupplier;


    public NetworkEvolutionManager(Fraction parentFraction, ParentSelector<E> parentSelector, Supplier<E> networkSupplier, Collection<OffspringGenerator<E>> offspringProviders) throws DimensionsMismatchException, IllegalArgumentException, NullPointerException{
        this(parentFraction, parentSelector, networkSupplier, offspringProviders, MultiPartRatio.uniform(offspringProviders.size()));
    }

    public NetworkEvolutionManager(Fraction parentFraction, ParentSelector<E> parentSelector, Supplier<E> networkSupplier, Collection<OffspringGenerator<E>> offspringProviders, MultiPartRatio distribution) throws DimensionsMismatchException, IllegalArgumentException, NullPointerException{
        if(offspringProviders.size() <= 0) throw new IllegalArgumentException("Illegal number of offspring providers. Collection.size() <= 0");

        this.networkSupplier = Objects.requireNonNull(networkSupplier, "Network supplier is null.");

        this.parentSelector = Objects.requireNonNull(parentSelector, "Parent selector is null.");
        
        networkDistribution = Objects.requireNonNull(distribution, "Distribution is null.");

        if(offspringProviders.size() != distribution.getNumTerms())
            throw new DimensionsMismatchException("Number of offspring providers does not match number of terms in the distribution.");

        this.parentFraction = Objects.requireNonNull(parentFraction, "Parent fraction is null.").floatValue();

        if(offspringProviders.stream().anyMatch(x -> x == null)) throw new NullPointerException("Offspring providers contains null.");

        this.offspringGenerators = Collections.unmodifiableList(new ArrayList<>(offspringProviders));

        expectedOffspring = 0;
        networksPerProvider = new int[offspringGenerators.size()];
    }

    public void setParallel(boolean enabled){
        if(enabled == parallel) return;
        parallel = enabled;
        
        updateNewGenerationFunction();
    }
    
    public void setGenerateMetadata(boolean enabled){
        if(enabled == createMetadata) return;
        createMetadata = enabled;
        updateNewGenerationFunction();
    }

    private void updateNewGenerationFunction(){
        synchronized(newGenerationFunction){
            newGenerationFunction = getNewGenerationFunction(parallel, createMetadata);
        }
    }

    private NewGenerationFunction<E> getNewGenerationFunction(boolean parallel, boolean withMetadata){
        if(parallel){
            return withMetadata ? new ParallelNewGenerationWithMetadata() : new ParallelNewGeneration();
        } else {
            return withMetadata ? this::linearNewGenerationWithMetadata : this::linearNewGeneration;
        }
    }

    public boolean isParallel() {
        return parallel;
    }

    public boolean isGeneratingMetadata() {
        return createMetadata;
    }

    public void setParentFraction(Fraction fraction) throws NullPointerException {
        this.parentFraction = fraction.floatValue();
    }

    public void setParentFraction(float fraction) throws IllegalArgumentException {
        if(fraction < 0 || fraction > 1) throw new IllegalArgumentException("Parent fraction must be in range [0, 1]");
        this.parentFraction = fraction;
    }

    public float getParentFraction() {
        return parentFraction;
    }

    public void setParentSelector(ParentSelector<E> selector) {
        this.parentSelector = selector;
    }

    public ParentSelector<E> getParentSelector() {
        return parentSelector;
    }

    public List<OffspringGenerator<E>> getOffspringGenerators() {
        return offspringGenerators;
    }


    /**
     * @param networkSupplier The supplier to use when creating child networks.
     * @implNote Changing the layout of networks supplied will likely lead to errors.
     */
    public void setNetworkSupplier(Supplier<E> networkSupplier) {
        this.networkSupplier = networkSupplier;
    }

    public List<ScoredNetwork<E>> createRandomGeneration(int numNetworks){
        ArrayList<ScoredNetwork<E>> networks = new ArrayList<>(numNetworks);
        NeuralNetworkTools.getRandomizedNetworks(numNetworks, networkSupplier).forEach(x -> networks.add(new ScoredNetwork<E>(x)));
        return networks;
    }

    public List<ScoredNetwork<E>> createNewGeneration(List<ScoredNetwork<E>> networks, int numOffspring){
        int numParents = Math.max(1, Math.round(networks.size() * parentFraction));
        List<ScoredNetwork<E>> parentNetworks = parentSelector.getParents(numParents, networks);

        updateNetworksPerProvider(numOffspring);

        synchronized(newGenerationFunction){
            return newGenerationFunction.apply(parentNetworks, numOffspring);
        }
    }

    private void updateNetworksPerProvider(int numOffspring){
        if(numOffspring == expectedOffspring)
            return;
        expectedOffspring = numOffspring;
        networksPerProvider = networkDistribution.distribute(expectedOffspring);
    }

    private List<ScoredNetwork<E>> createOffspringWithMetadata(List<ScoredNetwork<E>> parentNetworks, OffspringGenerator<E> generator, Integer numOffspring){
        List<ScoredNetwork<E>> offspring = generator.createOffspring(parentNetworks, numOffspring);
        
        for(ScoredNetwork<E> o : offspring)
            o.get().putMetadata("offspringGenerator", generator.getIdentifyer());

        return offspring;
    }

    private List<ScoredNetwork<E>> linearNewGenerationWithMetadata(List<ScoredNetwork<E>> parentNetworks, int numOffspring){
        ArrayList<ScoredNetwork<E>> newGeneration = new ArrayList<>(numOffspring);

        for(int i = 0; i < networksPerProvider.length; i++){
            OffspringGenerator<E> generator = offspringGenerators.get(i);
            List<ScoredNetwork<E>> offspring = createOffspringWithMetadata(parentNetworks, generator, networksPerProvider[i]);

            newGeneration.addAll(offspring);
        }

        return newGeneration;
    }

    private List<ScoredNetwork<E>> linearNewGeneration(List<ScoredNetwork<E>> parentNetworks, int numOffspring) {
        ArrayList<ScoredNetwork<E>> offspring = new ArrayList<>(numOffspring);

        for(int i = 0; i < networksPerProvider.length; i++)
            offspringGenerators.get(i).createOffspring(parentNetworks, networksPerProvider[i], offspring);

        return offspring;
    }

    private class ParallelNewGeneration implements NewGenerationFunction<E>{

        @Override
        public synchronized List<ScoredNetwork<E>> apply(List<ScoredNetwork<E>> parents, int numOffspring) {
            ArrayList<ScoredNetwork<E>> offspring = new ArrayList<>(numOffspring);
            ArrayList<ObjIntPair<OffspringGenerator<E>>> offspringCounts = new ArrayList<>(networksPerProvider.length);
            for (int i = 0; i < networksPerProvider.length; i++) {
                offspringCounts.add(new ObjIntPair<>(offspringGenerators.get(i), networksPerProvider[i]));
            }

            offspringCounts.parallelStream().forEach(x -> {
                List<ScoredNetwork<E>> children = createOffspring(parents, x.getObject(), x.getInteger());
                synchronized(offspring){
                    offspring.addAll(children);
                }
            });

            return offspring;
        }

        protected List<ScoredNetwork<E>> createOffspring(List<ScoredNetwork<E>> parents, OffspringGenerator<E> generator, int numOffspring){
            return generator.createOffspring(parents, numOffspring);
        }

    }

    private class ParallelNewGenerationWithMetadata extends ParallelNewGeneration {
        @Override
        protected List<ScoredNetwork<E>> createOffspring(List<ScoredNetwork<E>> parents, OffspringGenerator<E> generator, int numOffspring) {
            return createOffspringWithMetadata(parents, generator, numOffspring);
        }
    }

    @FunctionalInterface
    private interface NewGenerationFunction<E extends MutableNeuralNetwork> {
        public List<ScoredNetwork<E>> apply(List<ScoredNetwork<E>> parents, int numOffspring);
    }
}

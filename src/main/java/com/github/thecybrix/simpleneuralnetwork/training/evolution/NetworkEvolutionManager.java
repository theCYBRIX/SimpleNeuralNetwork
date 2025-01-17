package com.github.thecybrix.simpleneuralnetwork.training.evolution;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.function.Supplier;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkTools;
import com.github.thecybrix.simpleneuralnetwork.exceptions.DimensionsMismatchException;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.util.CompoundRatio;
import com.github.thecybrix.util.Fraction;

public class NetworkEvolutionManager<E extends MutableNeuralNetwork>{

    private static ExecutorService executorService;
    private static int executorServiceUsers = 0;

    final private CompoundRatio NETWORK_DISTRIBUTION;

    private boolean parallel = false, createMetadata = false;

    private BiFunction<List<ScoredNetwork<E>>, Integer, List<ScoredNetwork<E>>> newGenerationFunction = this::linearNewGeneration;
    private ParentSelector<E> parentSelector;

    private ArrayList<OffspringGenerator<E>> offspringGenerators;
    private float parentFraction;
    private Supplier<E> networkSupplier;


    public NetworkEvolutionManager(Fraction parentFraction, ParentSelector<E> parentSelector, Supplier<E> networkSupplier, Collection<OffspringGenerator<E>> offspringProviders) throws DimensionsMismatchException, IllegalArgumentException, ArithmeticException, NullPointerException{
        this(parentFraction, parentSelector, networkSupplier, offspringProviders, CompoundRatio.uniform(offspringProviders.size()));
    }

    public NetworkEvolutionManager(Fraction parentFraction, ParentSelector<E> parentSelector, Supplier<E> networkSupplier, Collection<OffspringGenerator<E>> offspringProviders, CompoundRatio distribution) throws DimensionsMismatchException, IllegalArgumentException, NullPointerException{
        if(offspringProviders.size() <= 0) throw new IllegalArgumentException("Illegal number of offspring providers. Collection.size() <= 0");

        this.networkSupplier = Objects.requireNonNull(networkSupplier, "Network supplier is null.");

        this.parentSelector = Objects.requireNonNull(parentSelector, "Parent selector is null.");
        
        NETWORK_DISTRIBUTION = Objects.requireNonNull(distribution, "Distribution is null.");

        if(offspringProviders.size() != distribution.getNumTerms())
            throw new DimensionsMismatchException("Number of offspring providers does not match number of terms in the distribution.");

        this.parentFraction = Objects.requireNonNull(parentFraction, "Parent fraction is null.").floatValue();

        if(offspringProviders.stream().anyMatch(x -> x == null)) throw new NullPointerException("Offspring providers contains null.");

        this.offspringGenerators = new ArrayList<>(offspringProviders);
    }


    private synchronized static void registerExecutorServiceUsage(){
        executorServiceUsers += 1;
        if(executorService == null) executorService = Executors.newWorkStealingPool();
    }

    private synchronized static void unregisterExecutorServiceUsage(){
        executorServiceUsers -= 1;
        if(executorServiceUsers != 0) return;

        try {
            executorService.shutdown();
            if (!executorService.awaitTermination(60, TimeUnit.SECONDS)) {
                executorService.shutdownNow();
            }
        } catch (InterruptedException e) {
            executorService.shutdownNow();
            Thread.currentThread().interrupt();
        } finally {
            executorService = null;
        }
    }


    public void setParallel(boolean enabled){
        if(enabled == parallel) return;
        parallel = enabled;

        if(parallel)
            registerExecutorServiceUsage();
        else
            unregisterExecutorServiceUsage();
        
        updateNewGenerationFunction();
    }
    
    public void setCreateMetadata(boolean enabled){
        if(enabled == createMetadata) return;
        createMetadata = enabled;
        updateNewGenerationFunction();
    }

    private void updateNewGenerationFunction(){
        synchronized(newGenerationFunction){
            newGenerationFunction = getNewGenerationFunction(parallel, createMetadata);
        }
    }

    private BiFunction<List<ScoredNetwork<E>>, Integer, List<ScoredNetwork<E>>> getNewGenerationFunction(boolean parallel, boolean withMetadata){
        if(parallel){
            return withMetadata ? new ParallelNewGenerationWithMetadata() : new ParallelNewGeneration();
        } else {
            return withMetadata ? this::linearNewGeneration : this::linearNewGenerationWithMetadata;
        }
    }

    public boolean isParallel() {
        return parallel;
    }

    public boolean isCreatingMetadata() {
        return createMetadata;
    }

    public void setParentFraction(Fraction fraction) throws NullPointerException {
        this.parentFraction = fraction.floatValue();
    }

    public void setParentFraction(float fraction) throws IllegalArgumentException {
        if(fraction < 0 || fraction > 1) throw new IllegalArgumentException("Parent fraction must be in range [0, 1]");
        this.parentFraction = fraction;
    }

    public void setParentSelector(ParentSelector<E> selector) {
        this.parentSelector = selector;
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

        synchronized(newGenerationFunction){
            return newGenerationFunction.apply(parentNetworks, numOffspring);
        }
    }

    private Integer[] getNumNetworksPerProvider(int totalNumNetworks){
        Integer[] nerworksPerProvider = new Integer[NETWORK_DISTRIBUTION.getNumTerms()];
        for(int i = 0; i < nerworksPerProvider.length; i++)
            nerworksPerProvider[i] = Integer.valueOf(Math.round(totalNumNetworks * NETWORK_DISTRIBUTION.getFraction(i).floatValue()));
        return nerworksPerProvider;
    }

    private List<ScoredNetwork<E>> createOffspringWithMetadata(List<ScoredNetwork<E>> parentNetworks, OffspringGenerator<E> generator, Integer numOffspring){
        List<ScoredNetwork<E>> offspring = generator.createOffspring(parentNetworks, numOffspring);
        
        for(ScoredNetwork<E> o : offspring)
            o.get().putMetadata("offspringGenerator", generator.getIdentifyer());

        return offspring;
    }

    private List<ScoredNetwork<E>> linearNewGenerationWithMetadata(List<ScoredNetwork<E>> parentNetworks, Integer numOffspring){
        ArrayList<ScoredNetwork<E>> newGeneration = new ArrayList<>(numOffspring);
        Integer[] networksPerProvider = getNumNetworksPerProvider(numOffspring);

        for(int i = 0; i < networksPerProvider.length; i++){
            OffspringGenerator<E> generator = offspringGenerators.get(i);
            List<ScoredNetwork<E>> offspring = createOffspringWithMetadata(parentNetworks, generator, networksPerProvider[i]);

            offspring.addAll(offspring);
        }

        return newGeneration;
    }

    private List<ScoredNetwork<E>> linearNewGeneration(List<ScoredNetwork<E>> parentNetworks, Integer numOffspring) {
        ArrayList<ScoredNetwork<E>> offspring = new ArrayList<>(numOffspring);
        Integer[] networksPerProvider = getNumNetworksPerProvider(numOffspring);

        for(int i = 0; i < networksPerProvider.length; i++)
            offspring.addAll(offspringGenerators.get(i).createOffspring(parentNetworks, networksPerProvider[i]));

        return offspring;
    }

    private class ParallelNewGeneration implements BiFunction<List<ScoredNetwork<E>>, Integer, List<ScoredNetwork<E>>>{

        @Override
        public List<ScoredNetwork<E>> apply(List<ScoredNetwork<E>> parents, Integer numOffspring) {
            ArrayList<ScoredNetwork<E>> offspring = new ArrayList<>(numOffspring);
            ArrayList<Future<List<ScoredNetwork<E>>>> processes = new ArrayList<>(offspringGenerators.size());
            Integer[] networksPerProvider = getNumNetworksPerProvider(numOffspring);

            for(int i = 0; i < offspringGenerators.size(); i++){
                final int index = i;
                Callable<List<ScoredNetwork<E>>> task = () -> createOffspring(parents, offspringGenerators.get(index), networksPerProvider[index]);
                processes.add(executorService.submit(task));
            }

            for(Future<List<ScoredNetwork<E>>> future : processes)
                try {
                    offspring.addAll(future.get());
                } catch(InterruptedException | CancellationException e) {
                    break;
                } catch (ExecutionException e) {
                    e.printStackTrace();
                }

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
}

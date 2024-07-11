package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.BiFunction;
import java.util.function.Supplier;

import org.apache.commons.numbers.fraction.Fraction;

import com.mjsd.simpleneuralnetwork.NeuralNetworkTools;
import com.mjsd.simpleneuralnetwork.exceptions.DimensionsMismatchException;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.ScoredNetwork;

public class NetworkEvolutionManager<E extends MutableNeuralNetwork, T extends Comparable<T>>{
    final public Supplier<E> NETWORK_SUPPLIER;
    final private CompoundRatio NETWORK_DISTRIBUTION;
    private boolean parallel = false;
    private BiFunction<List<ScoredNetwork<E, T>>, Integer, Collection<ScoredNetwork<E, T>>> newGenerationAlgorithm = this::linearNewGeneration;
    private ArrayList<OffspringGenerator<E, T>> offspringGenerators;
    private float parentFraction;

    public NetworkEvolutionManager(Fraction parentFraction, Supplier<E> networkSupplier, Collection<OffspringGenerator<E, T>> offspringProviders, CompoundRatio distribution) throws DimensionsMismatchException, IllegalArgumentException, NullPointerException{
        if(offspringProviders.size() <= 0) throw new IllegalArgumentException("Illegal number of offspring providers. Collection.size() <= 0");

        NETWORK_SUPPLIER = Objects.requireNonNull(networkSupplier, "Network supplier is null.");
        
        NETWORK_DISTRIBUTION = Objects.requireNonNull(distribution, "Distribution is null.");

        if(offspringProviders.size() != distribution.getNumTerms())
            throw new DimensionsMismatchException("Number of offspring providers does not match number of terms in the distribution.");

        this.parentFraction = Objects.requireNonNull(parentFraction, "Parent fraction is null.").floatValue();

        if(offspringProviders.stream().anyMatch(x -> x == null)) throw new NullPointerException("Offspring providers contains null.");

        this.offspringGenerators = new ArrayList<>(offspringProviders);
    }

    public NetworkEvolutionManager(Fraction parentFraction, Supplier<E> networkSupplier, Collection<OffspringGenerator<E, T>> offspringProviders) throws DimensionsMismatchException, IllegalArgumentException, ArithmeticException, NullPointerException{
        this(parentFraction, networkSupplier, offspringProviders, CompoundRatio.uniform(offspringProviders.size()));
    }

    public void setParallel(boolean enabled){
        if(enabled == parallel) return;
        synchronized(newGenerationAlgorithm){
            parallel = enabled;
            newGenerationAlgorithm = parallel ? new ParallelNewGeneration() : this::linearNewGeneration;
        }
    }

    public boolean isParallel() {
        return parallel;
    }

    public void setParentFraction(Fraction fraction) throws NullPointerException {
        this.parentFraction = fraction.floatValue();
    }

    public void setParentFraction(float fraction) throws IllegalArgumentException {
        if(fraction < 0 || fraction > 1) throw new IllegalArgumentException("Parent fraction must be in range [0, 1]");
        this.parentFraction = fraction;
    }

    public Collection<ScoredNetwork<E, T>> createRandomGeneration(int numNetworks){
        ArrayList<ScoredNetwork<E, T>> networks = new ArrayList<>(numNetworks);
        NeuralNetworkTools.getRandomizedNetworks(numNetworks, NETWORK_SUPPLIER).forEach(x -> networks.add(new ScoredNetwork<E, T>(x)));
        return networks;
    }

    public Collection<ScoredNetwork<E, T>> createNewGeneration(List<ScoredNetwork<E, T>> networks, int numOffspring){
        synchronized(newGenerationAlgorithm){
            Collections.sort(networks);

            List<ScoredNetwork<E, T>> parentNetworks = networks.subList(Math.round(networks.size() * (1 - parentFraction)), networks.size());

            return newGenerationAlgorithm.apply(parentNetworks, numOffspring);
        }
    }

    public Collection<ScoredNetwork<E, T>> createNewGeneration(List<ScoredNetwork<E, T>> networks, int numOffspring, Comparator<ScoredNetwork<E, T>> comparator){
        synchronized(newGenerationAlgorithm){
            Collections.sort(networks, comparator);

            List<ScoredNetwork<E, T>> parentNetworks = networks.subList(Math.min(Math.round(networks.size() * (1 - parentFraction)), networks.size() - 1), networks.size());

            return newGenerationAlgorithm.apply(parentNetworks, numOffspring);
        }
    }

    private Integer[] getNumNetworksPerProvider(int totalNumNetworks){
        Integer[] nerworksPerProvider = new Integer[NETWORK_DISTRIBUTION.getNumTerms()];
        for(int i = 0; i < nerworksPerProvider.length; i++)
            nerworksPerProvider[i] = Integer.valueOf(Math.round(totalNumNetworks * NETWORK_DISTRIBUTION.getFraction(i).floatValue()));
        return nerworksPerProvider;
    }

    private List<ScoredNetwork<E, T>> linearNewGeneration(List<ScoredNetwork<E, T>> networks, Integer numOffspring) {
        ArrayList<ScoredNetwork<E, T>> offspring = new ArrayList<>(numOffspring);
        Integer[] networksPerProvider = getNumNetworksPerProvider(numOffspring);

        for(int i = 0; i < networksPerProvider.length; i++)
            offspring.addAll(offspringGenerators.get(i).createOffspring(networks, networksPerProvider[i]));

        return offspring;
    }

    private class ParallelNewGeneration implements BiFunction<List<ScoredNetwork<E, T>>, Integer, Collection<ScoredNetwork<E, T>>>{
        ExecutorService executorService = Executors.newWorkStealingPool();

        @Override
        public Collection<ScoredNetwork<E, T>> apply(List<ScoredNetwork<E, T>> parents, Integer numOffspring) {
            ArrayList<ScoredNetwork<E, T>> offspring = new ArrayList<>(numOffspring);
            ArrayList<Future<Collection<ScoredNetwork<E, T>>>> processes = new ArrayList<>(offspringGenerators.size());
            Integer[] networksPerProvider = getNumNetworksPerProvider(numOffspring);

            for(int i = 0; i < offspringGenerators.size(); i++){
                final OffspringGenerator<E, T> PROVIDER = offspringGenerators.get(i);
                final int NUM_OFFSPRING = networksPerProvider[i];
                Callable<Collection<ScoredNetwork<E, T>>> task = () -> PROVIDER.createOffspring(parents, NUM_OFFSPRING);
                processes.add(executorService.submit(task));
            }

            for(Future<Collection<ScoredNetwork<E, T>>> future : processes)
                try {
                    offspring.addAll(future.get());
                } catch(InterruptedException | CancellationException e) {
                    break;
                } catch (ExecutionException e) {
                    e.printStackTrace();
                }

            return offspring;
        }

    }
}

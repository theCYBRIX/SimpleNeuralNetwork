package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Supplier;

import org.apache.commons.numbers.fraction.Fraction;

import com.mjsd.simpleneuralnetwork.NeuralNetworkTools;
import com.mjsd.simpleneuralnetwork.exceptions.DimensionsMismatchException;
import com.mjsd.simpleneuralnetwork.training.RankedNeuralNetwork;

public class Population<E extends RankedNeuralNetwork> {
    final public Supplier<E> NETWORK_PROVIDER;
    final private CompoundRatio NETWORK_DISTRIBUTION;
    final private Integer[] NETWORKS_PER_PROVIDER;
    final private Comparator<E> COMPARATOR;
    private boolean parallel = false;
    private Runnable newGenerationAlgorithm = this::linearNewGeneration;
    protected ArrayList<E> currentGeneration = new ArrayList<>();
    private ArrayList<E> leaderBoard = new ArrayList<>();
    private List<E> parentNetworks;
    private ArrayList<OffspringProvider<E>> offspringProviders;
    private int totalNumNetworks, numParentNetworks;
    private float parentFraction;
    private ArrayList<E> previousLeaderBoard = null; 
    private LinkedList<E> manualAdditions = new LinkedList<>(); 

    private Optional<Double> bestScore = Optional.empty();

    public Population(int numNetworks, Fraction parentFraction, Supplier<E> networkProvider, Collection<OffspringProvider<E>> offspringProviders, CompoundRatio distribution, Comparator<E> comparator) throws DimensionsMismatchException, IllegalArgumentException, NullPointerException{
        if(offspringProviders.size() <= 0) throw new IllegalArgumentException("Illegal number of offspring providers. Collection.size() <= 0");
        if(numNetworks < 0) throw new IllegalArgumentException("Invalid num networks. (" + numNetworks + " < 0)");

        NETWORK_PROVIDER = Objects.requireNonNull(networkProvider, "Network provider is null.");
        
        NETWORK_DISTRIBUTION = Objects.requireNonNull(distribution, "Distribution is null.");

        if(offspringProviders.size() != distribution.getNumTerms())
            throw new DimensionsMismatchException("Number of offspring providers does not match number of terms in the distribution.");

        this.parentFraction = Objects.requireNonNull(parentFraction, "Parent fraction is null.").floatValue();

        COMPARATOR = Objects.requireNonNull(comparator, "Comparator is null.");

        if(offspringProviders.stream().anyMatch(x -> x == null)) throw new NullPointerException("Offspring providers contains null.");

        this.offspringProviders = new ArrayList<>(offspringProviders);

        NETWORKS_PER_PROVIDER = new Integer[offspringProviders.size()];
        
        totalNumNetworks = numNetworks;
        updateNumNetworksPerProvider();

        currentGeneration.ensureCapacity(totalNumNetworks);
        leaderBoard.ensureCapacity(totalNumNetworks);
    }

    public Population(Fraction parentFraction, Supplier<E> networkProvider, Collection<OffspringProvider<E>> offspringProviders, CompoundRatio distribution, Comparator<E> comparator) throws DimensionsMismatchException, IllegalArgumentException, ArithmeticException, NullPointerException{
        this(1, parentFraction, networkProvider, offspringProviders, distribution, comparator);
    }

    public Population(Fraction parentFraction, Supplier<E> networkProvider, Collection<OffspringProvider<E>> offspringProviders, Comparator<E> comparator) throws DimensionsMismatchException, IllegalArgumentException, ArithmeticException, NullPointerException{
        this(1, parentFraction, networkProvider, offspringProviders, CompoundRatio.uniform(offspringProviders.size()), comparator);
    }

    public Population(int numNetworks, int numParentNetworks, Supplier<E> networkProvider, Collection<OffspringProvider<E>> offspringProviders, CompoundRatio distribution, Comparator<E> comparator) throws DimensionsMismatchException, IllegalArgumentException, ArithmeticException, NullPointerException{
        this(numNetworks, Fraction.of(numNetworks, numParentNetworks), networkProvider, offspringProviders, distribution, comparator);
    }

    public Population(int numNetworks, int numParentNetworks, Supplier<E> networkProvider, Collection<OffspringProvider<E>> offspringProviders, Comparator<E> comparator) throws IllegalArgumentException, NullPointerException{
        this(numNetworks, numParentNetworks, networkProvider, offspringProviders, CompoundRatio.uniform(offspringProviders.size()), comparator);
    }

    public ArrayList<E> getCurrentGeneration() {
        return new ArrayList<>(currentGeneration);
    }

    public Optional<Double> getBestScore() {
        return bestScore;
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

    public void setNumNetworks(int numNetworks) {
        this.totalNumNetworks = numNetworks;
        updateNumParentNetworks();
        updateNumNetworksPerProvider();
        ensureSufficientNetworks();
    }

    public void setNumNetworks(int numNetworks, int numParentNetworks) {
        this.totalNumNetworks = numNetworks;
        setParentFraction(Fraction.of(numParentNetworks, numNetworks));
        updateNumNetworksPerProvider();
        ensureSufficientNetworks();
    }

    public void setParentFraction(Fraction parentFraction) throws NullPointerException {
        this.parentFraction = parentFraction.floatValue();
        updateNumParentNetworks();
    }

    public void add(E network) throws NullPointerException{
        addUnchecked(Objects.requireNonNull(network, "Network is null."));
    }

    protected void addUnchecked(E network){
        synchronized(newGenerationAlgorithm){
            manualAdditions.add(network);
        }
    }

    public void addAll(Collection<? extends E> networks) throws NullPointerException{
        Objects.requireNonNull(networks, "Collection is null.");
        if((parallel ? networks.parallelStream() : networks.stream()).anyMatch(x -> x == null)) throw new NullPointerException("Collection contains null.");
        addAllUnchecked(networks);
    }

protected void addAllUnchecked(Collection<? extends E> networks){
        synchronized(newGenerationAlgorithm){
            manualAdditions.addAll(networks);
        }
    }

    protected void ensureSufficientNetworks(){
        synchronized(newGenerationAlgorithm){
            while(manualAdditions.size() > 0)
                currentGeneration.add(manualAdditions.removeFirst());
            
            if(currentGeneration.size() < totalNumNetworks)
                currentGeneration.addAll(NeuralNetworkTools.getRandomizedNetworks(totalNumNetworks - currentGeneration.size(), NETWORK_PROVIDER));
        }
    }

    public void populateNewGeneration(){
        synchronized(newGenerationAlgorithm){
            ArrayList<E> temp = leaderBoard;
            leaderBoard = currentGeneration;
            currentGeneration = temp;

            Collections.sort(leaderBoard, COMPARATOR);
            previousLeaderBoard = new ArrayList<E>(leaderBoard);

            int lastIndex = leaderBoard.size() - 1;
            bestScore = lastIndex >= 0 ? leaderBoard.get(lastIndex).getScore() : Optional.empty();

            parentNetworks = leaderBoard.subList(Math.max(0, leaderBoard.size() - numParentNetworks), leaderBoard.size());

            newGenerationAlgorithm.run();

            if(manualAdditions.isEmpty()) return;

            do{
                currentGeneration.add(manualAdditions.removeFirst());
            } while(manualAdditions.size() > 0);
        }
    }

    public ArrayList<E> getPreviousLeaderBoard() {
        return previousLeaderBoard;
    }

    private void updateNumNetworksPerProvider(){
        for(int i = 0; i < NETWORKS_PER_PROVIDER.length; i++)
            NETWORKS_PER_PROVIDER[i] = Integer.valueOf(Math.round(totalNumNetworks * NETWORK_DISTRIBUTION.getFraction(i).floatValue()));
    }

    private void updateNumParentNetworks() {
        this.numParentNetworks = Math.round(totalNumNetworks * parentFraction);
    }

    private void linearNewGeneration() {
        synchronized(currentGeneration){
            currentGeneration.clear();

            for(int i = 0; i < NETWORKS_PER_PROVIDER.length; i++)
                currentGeneration.addAll(offspringProviders.get(i).createOffspring(parentNetworks, NETWORKS_PER_PROVIDER[i]));
        }
    }

    private class ParallelNewGeneration implements Runnable{
        ExecutorService executorService = Executors.newCachedThreadPool();
        ArrayList<Future<Collection<E>>> processes = new ArrayList<>(offspringProviders.size());

        @Override
        public void run() {
            processes.clear();

            for(int i = 0; i < offspringProviders.size(); i++){
                OffspringProvider<E> provider = offspringProviders.get(i);
                int numOffspring = NETWORKS_PER_PROVIDER[i];
                Callable<Collection<E>> task = () -> provider.createOffspring(parentNetworks, numOffspring);
                processes.add(executorService.submit(task));
            }

            synchronized(currentGeneration){
                currentGeneration.clear();

                for(Future<Collection<E>> future : processes)
                    try {
                        currentGeneration.addAll(future.get()); 
                    } catch(InterruptedException e) {
                        break;
                    } catch (Exception e) {
                        continue;
                    }
            }
        }

    }
}

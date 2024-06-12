package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Supplier;
import java.util.stream.Stream;

import com.mjsd.simpleneuralnetwork.exceptions.IllegalTermCountException;
import com.mjsd.simpleneuralnetwork.training.RankedNeuralNetwork;

public class Ecosystem<E extends RankedNeuralNetwork> implements Population<E> {

    final private List<Population<E>> POPULATIONS;
    final private int[] POPULATION_SIZES;
    private float[] networkDistribution;
    private int totalNumNetworks;
    private Runnable newGenerationAlgorithm = this::linearNewGeneration;
    private boolean parallel = false;
    private Supplier<Stream<Population<E>>> populationStream;
    
    public Ecosystem(int totalNumNetworks, Collection<? extends Population<E>> populations, CompoundRatio networkDistribution) throws IllegalArgumentException, NullPointerException {
        if(totalNumNetworks < 0) throw new IllegalArgumentException("Illegal number of networks: " + totalNumNetworks);

        if(populations.stream().anyMatch(x -> x == null)) throw new NullPointerException("Collection of populations contains null.");

        this.totalNumNetworks = totalNumNetworks;

        POPULATIONS = Collections.unmodifiableList(new ArrayList<>(populations));
        populationStream = POPULATIONS::stream;

        POPULATION_SIZES = new int[POPULATIONS.size()];

        setNetworkDistribution(networkDistribution);
        setPopulationsParallel(parallel);
    }
    
    public Ecosystem(int totalNumNetworks, Collection<? extends Population<E>> populations) throws IllegalArgumentException, NullPointerException{
        this(totalNumNetworks, populations, CompoundRatio.uniform(populations.size()));
    }
    
    public Ecosystem(int totalNumNetworks, Population<E> population) throws IllegalArgumentException, NullPointerException{
        if(totalNumNetworks < 0) throw new IllegalArgumentException("Illegal total number of networks: " + totalNumNetworks);

        Objects.requireNonNull(population, "Population is null.");

        this.totalNumNetworks = totalNumNetworks;
        population.setSize(totalNumNetworks);

        POPULATIONS = Collections.unmodifiableList(Arrays.asList(population));
        populationStream = POPULATIONS::stream;
        networkDistribution = new float[]{ 1.0f };

        POPULATION_SIZES = new int[]{ totalNumNetworks };
        
        setPopulationsParallel(parallel);
    }

    public int size() {
        return totalNumNetworks;
    }

    public List<Population<E>> getPopulations() {
        return POPULATIONS;
    }

    public Optional<Double> getBestScore(){
        return populationStream.get()
               .map(x -> x.getBestScore())
               .filter(x -> x.isPresent())
               .map(x -> x.get())
               .max((x, y) -> x.compareTo(y));
    }

    public List<E> getLeaderBoard(){
        return getLeaderBoard(null);
    }

    public ArrayList<E> getLeaderBoard(Comparator<E> comparator){
        ArrayList<E> leaderBoard = new ArrayList<>(totalNumNetworks);
        populationStream.get()
                        .map(x -> x.getLeaderBoard())
                        .flatMap(x -> x.stream())
                        .sorted(comparator == null ? (x, y) -> x.compareTo(y) : comparator)
                        .forEachOrdered(x -> leaderBoard.add(x));
        return leaderBoard;
    }

    public ArrayList<E> getMembers(){
        ArrayList<E> networks = new ArrayList<>(totalNumNetworks);

        for(Population<E> population : POPULATIONS)
            networks.addAll(population.getMembers());

        return networks;
    }

    public void add(E network){
        Objects.requireNonNull(network, "Network is null.");
        populationStream.get().forEach(x -> x.add(network));
    }

    public void addAll(Collection<? extends E> networks){
        Objects.requireNonNull(networks, "Collection is null.");
        if((parallel ? networks.parallelStream() : networks.stream()).anyMatch(x -> x == null))
            Objects.requireNonNull(networks, "Collection contains null.");

        populationStream.get().forEach(x -> x.addAll(networks));
    }

    public void setSize(int totalNumNetworks) throws IllegalArgumentException {
        if(totalNumNetworks < 0) throw new IllegalArgumentException("Illegal total number of networks: " + totalNumNetworks);
        this.totalNumNetworks = totalNumNetworks;
        updatePopulationSizes();
    }

    public void setNetworkDistribution(CompoundRatio networkDistribution) throws IllegalArgumentException {
        Ecosystem.validateDistribution(POPULATIONS, networkDistribution);

        this.networkDistribution = new float[networkDistribution.getNumTerms()];
            
        for(int i = 0; i < networkDistribution.getNumTerms(); i++)
            this.networkDistribution[i] = networkDistribution.getFraction(i).floatValue();

        updatePopulationSizes();
    }

    private void updatePopulationSizes(){
        for(int i = 0; i < POPULATIONS.size(); i++){
            POPULATION_SIZES[i] = Math.round(totalNumNetworks * networkDistribution[i]);
            POPULATIONS.get(i).setSize(POPULATION_SIZES[i]);
        }
    }

    public void populateNewGeneration(){
        synchronized(newGenerationAlgorithm){
            newGenerationAlgorithm.run();
        }
    }

    public void setParallel(boolean enabled) {
        if(parallel == enabled) return;
        synchronized(newGenerationAlgorithm){
            this.parallel = enabled;

            if(parallel){
                newGenerationAlgorithm = new ParallelNewGeneration();
                populationStream = POPULATIONS::parallelStream;
            } else {
                newGenerationAlgorithm = this::linearNewGeneration;
                populationStream = POPULATIONS::stream;
            }
            setPopulationsParallel(parallel);
        }
    }

    private void setPopulationsParallel(boolean enabled) {
        populationStream.get().forEach(x -> x.setParallel(enabled));
    }

    

    public boolean isParallel() {
        return parallel;
    }

    public void ensureSufficientNetworks(){
        populationStream.get().forEach(x -> x.ensureSufficientNetworks());
    }

    private static <E extends RankedNeuralNetwork> void validateDistribution(Collection<Population<E>> populations, CompoundRatio distribution) throws IllegalTermCountException, NullPointerException{
        CompoundRatio.requireNumberOfTerms(distribution, populations.size(), "Number of ratio terms does not match number of populations. " + distribution.getNumTerms() + " != " + populations.size());
    }

    private void linearNewGeneration(){
        for(Population<E> population : POPULATIONS)
            population.populateNewGeneration();
    }

    private class ParallelNewGeneration implements Runnable{
        ExecutorService executorService = Executors.newCachedThreadPool();
        ArrayList<Future<?>> processes = new ArrayList<>(POPULATIONS.size());

        @Override
        public void run() {
            processes.clear();

            for(int i = 0; i < POPULATIONS.size(); i++){
                Population<E> population = POPULATIONS.get(i);
                processes.add(executorService.submit(() -> population.populateNewGeneration()));
            }

            for(Future<?> future : processes)
                try {
                    future.get(); 
                } catch(InterruptedException e) {
                    break;
                } catch (Exception e) {
                    continue;
                }
        }

    }
    
}

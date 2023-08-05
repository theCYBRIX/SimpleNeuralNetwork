package mjsd.simpleneuralnetwork.training.evolution;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import mjsd.simpleneuralnetwork.exceptions.IllegalTermCountException;
import mjsd.simpleneuralnetwork.training.RankedNeuralNetwork;

public class Ecosystem<E extends RankedNeuralNetwork> {

    final private List<Population<E>> POPULATIONS;
    final private int[] POPULATION_SIZES;
    private float[] networkDistribution;
    private int totalNumNetworks;
    private Runnable newGenerationAlgorithm = this::linearNewGeneration;
    private boolean parallel = false;
    
    public Ecosystem(int totalNumNetworks, Collection<Population<E>> populations, CompoundRatio networkDistribution) throws IllegalArgumentException, NullPointerException {
        if(totalNumNetworks < 0) throw new IllegalArgumentException("Illegal number of networks: " + totalNumNetworks);

        if(populations.stream().anyMatch(x -> x == null)) throw new NullPointerException("Collection of populations contains null.");

        this.totalNumNetworks = totalNumNetworks;

        POPULATIONS = Collections.unmodifiableList(new ArrayList<>(populations));

        POPULATION_SIZES = new int[POPULATIONS.size()];

        setNetworkDistribution(networkDistribution);
        setPopulationsParallel(parallel);
    }
    
    public Ecosystem(int totalNumNetworks, Collection<Population<E>> populations) throws IllegalArgumentException, NullPointerException{
        this(totalNumNetworks, populations, CompoundRatio.uniform(populations.size()));
    }
    
    public Ecosystem(int totalNumNetworks, Population<E> population) throws IllegalArgumentException, NullPointerException{
        if(totalNumNetworks < 0) throw new IllegalArgumentException("Illegal total number of networks: " + totalNumNetworks);

        Objects.requireNonNull(population, "Population is null.");

        this.totalNumNetworks = totalNumNetworks;
        population.setNumNetworks(totalNumNetworks);

        POPULATIONS = Collections.unmodifiableList(Arrays.asList(population));
        networkDistribution = new float[]{ 1.0f };

        POPULATION_SIZES = new int[]{ totalNumNetworks };
        
        setPopulationsParallel(parallel);
    }

    public int getNumNetworks() {
        return totalNumNetworks;
    }

    public List<Population<E>> getPopulations() {
        return POPULATIONS;
    }

    public Optional<Double> getBestScore(){
        return POPULATIONS.stream()
               .map(x -> x.getBestScore())
               .filter(x -> x.isPresent())
               .map(x -> x.get())
               .max((x, y) -> x.compareTo(y));
    }

    public ArrayList<E> getCurrentGeneration(){
        ArrayList<E> networks = new ArrayList<>(totalNumNetworks);

        for(Population<E> population : POPULATIONS)
            networks.addAll(population.currentGeneration);

        return networks;
    }

    public void setNumNetworks(int totalNumNetworks) throws IllegalArgumentException {
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
            POPULATIONS.get(i).setNumNetworks(POPULATION_SIZES[i]);
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
            newGenerationAlgorithm = parallel ? new ParallelNewGeneration() : this::linearNewGeneration;

            setPopulationsParallel(parallel);
        }
    }

    private void setPopulationsParallel(boolean enabled) {
        for(Population<E> population : POPULATIONS)
            population.setParallel(enabled);
    }

    

    public boolean isParallel() {
        return parallel;
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

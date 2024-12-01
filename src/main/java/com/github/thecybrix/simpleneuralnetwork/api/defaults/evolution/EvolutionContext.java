package com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.stream.Collectors;

import com.github.thecybrix.simpleneuralnetwork.api.RequestHandler;
import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils.ParentSelection;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayout;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkBuilder;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkTools;
import com.github.thecybrix.simpleneuralnetwork.exceptions.DimensionsMismatchException;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.EvolutionaryTrainer;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.NetworkEvolutionManager;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelector;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ValueMappingTrainer;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.simple.SimpleEvolutionManager;
import com.github.thecybrix.util.IDManager;

public class EvolutionContext<E extends MutableNeuralNetwork> {

    public static enum State {
        RUNNING,
        SUCCESS,
        CANCELLED,
        FAILED
    }
    
    final private ExecutorService EXECUTOR_SERVICE;
    final private NeuralNetworkBuilder<E> NETWORK_BUILDER;

    final private Object PREV_GEN_LOCK = new Object();
    final private Object CURRENT_GEN_LOCK = new Object();

    private Map<Integer, ScoredNetwork<E>> previousGeneration = Collections.emptyMap();

    private NetworkEvolutionManager<E> evolutionManager;
    private ParentSelector<E> parentSelector;

    private Future<?> dataTrainedNetworks;
    private EvolutionaryTrainer<E> datasetTrainer;
    private int numTrainingSamples;
    private Instant trainingStartTime, trainingEndTime;
    
    private int numNetworks;
    private IDManager idManager = new IDManager();
    private HashMap<Integer, ScoredNetwork<E>> neuralNetworks = new HashMap<>();

    public EvolutionContext(NeuralNetworkBuilder<E> networkBuilder, ParentSelector<E> parentSelector, ExecutorService executorService){
        NETWORK_BUILDER = Objects.requireNonNull(networkBuilder, "Network builder is null.");
        EXECUTOR_SERVICE = (executorService != null) ? executorService : Executors.newWorkStealingPool();
        this.parentSelector = (parentSelector != null) ? parentSelector : ParentSelector.eliteSelection();
    }

    //TODO: make redundant or inferable information optional
    public void setup(int numNetworks, NetworkLayout layout, ParentSelection parentSelection, List<MutableNeuralNetwork> initialNetworks) throws NoSuchElementException, DimensionsMismatchException, NullPointerException {

        if(initialNetworks != null)
            NeuralNetworkTools.requireSameDimensions(initialNetworks);

        parentSelector = RequestHandlerUtils.getParentSelector(parentSelection);
        
        NETWORK_BUILDER.reset().withLayout(layout);
        evolutionManager = new SimpleEvolutionManager<>(NETWORK_BUILDER::build, parentSelector);
        evolutionManager.setCreateMetadata(true);

        this.numNetworks = numNetworks;

        synchronized(PREV_GEN_LOCK){
            if(previousGeneration.size() > 0)
                setPrevGen(Collections.emptyMap());
        }

        synchronized(CURRENT_GEN_LOCK){
            if(initialNetworks != null)
                addNetworks(
                    initialNetworks.parallelStream()
                    .map(x -> new ScoredNetwork<E>(NETWORK_BUILDER.convert(x)))
                    .collect(Collectors.toList())
                );
                
            addNetworks(evolutionManager.createRandomGeneration(numNetworks - neuralNetworks.size()));
        }
        
    }

    public void addNetworks(List<ScoredNetwork<E>> networks){
        for (ScoredNetwork<E> n : networks) 
            neuralNetworks.put(idManager.getNextID(), n);
    }

    public Map<Integer, double[]> processInputs(Map<Integer, double[]> inputData){

        ArrayList<Callable<Void>> tasks = new ArrayList<>(inputData.size());
        Map<Integer, double[]> outputs = new HashMap<Integer, double[]>(inputData.size());
        
        synchronized(CURRENT_GEN_LOCK){
            for (Entry<Integer, double[]> item : inputData.entrySet()) {
                tasks.add(() -> { process(item.getKey(), item.getValue(), outputs); return null; });
            }

            try {
                EXECUTOR_SERVICE.invokeAll(tasks);
            } catch (InterruptedException e) {
            } catch (NullPointerException | RejectedExecutionException e) {
                e.printStackTrace();
            }
        }

        return outputs;
    }

    private void process(int networkIndex, double[] inputs, Map<Integer, double[]> results) throws DimensionsMismatchException, NullPointerException{
        E network = neuralNetworks.get(networkIndex).get();
        synchronized(network){
            network.setInputs(inputs);
            network.forwardPass();
            synchronized(results){
                results.put(networkIndex, network.getOutputs());
            }
        }
    }

    public void approximateDataSet(TrainingDataSet dataSet) throws IllegalArgumentException, InterruptedException, ExecutionException, NullPointerException{
        if(datasetTrainer != null) stopTraining();
        
        ValueMappingTrainer<E> trainingScenario = ValueMappingTrainer.of(dataSet.inputs, dataSet.outputs, new ValueMappingTrainer.MeanSquaredError(), EXECUTOR_SERVICE);
        datasetTrainer = new EvolutionaryTrainer<>(numNetworks, evolutionManager, trainingScenario);
        datasetTrainer.attachCallback(trainer -> {
            setCurrentGen(trainer.getNetworks());
        });
        datasetTrainer.addAllScored(neuralNetworks.values());
        numTrainingSamples = dataSet.inputs.length;
        trainingEndTime = null;
        dataTrainedNetworks = EXECUTOR_SERVICE.submit(() -> {
            datasetTrainer.run();
            trainingEndTime = Instant.now();
            setCurrentGen(new ArrayList<>(datasetTrainer.getNetworks()));
        });
        trainingStartTime = Instant.now();
    }

    public void stopTraining() throws IllegalStateException, InterruptedException, ExecutionException {
        if(datasetTrainer == null) throw new IllegalStateException("Trainer has not been initialized.");
        if(datasetTrainer.isRunning()){
            datasetTrainer.stop();
            dataTrainedNetworks.get();
        }
    }

    public void createNewGeneration(HashMap<Integer, Double> scores) throws NoSuchElementException, NullPointerException {
        synchronized(CURRENT_GEN_LOCK){
            
            @SuppressWarnings("unchecked")
            ScoredNetwork<E>[] selectedNetworks = (ScoredNetwork<E>[]) new ScoredNetwork[scores.size()];
            double[] networkScores = new double[scores.size()];
            int selectionIndex = 0;
            Iterator<Entry<Integer, Double>> entrySet = scores.entrySet().iterator();
            Entry<Integer, Double> entry = null;
            boolean invalidEntry = false;

            while (entrySet.hasNext()){
                entry = entrySet.next();
                Double value = entry.getValue();
                ScoredNetwork<E> network = neuralNetworks.get(entry.getKey());

                if(network == null || value == null){
                    invalidEntry = true;
                    break;
                }

                selectedNetworks[selectionIndex] = network;
                networkScores[selectionIndex] = value;
                selectionIndex++;
            }

            if (invalidEntry){
                ArrayList<Entry<Integer, Double>> invalidEntries = new ArrayList<>();
                invalidEntries.add(entry);
                
                while (entrySet.hasNext()){
                    entry = entrySet.next();
                    Double value = entry.getValue();
                    ScoredNetwork<E> network = neuralNetworks.get(entry.getKey());

                    if(network == null || value == null)
                        invalidEntries.add(entry);
                }

                StringBuilder errorMessage = new StringBuilder("Invalid network ID");
                if(invalidEntries.size() > 1) errorMessage.append("s");
                errorMessage.append(" provided. (ID, value)");

                for (int i = 0; i < networkScores.length; i++) {
                    entry = invalidEntries.get(i);
                    errorMessage.append("\n(")
                                .append(entry.getKey())
                                .append(", ")
                                .append(entry.getValue())
                                .append(")");
                }
                throw new NoSuchElementException(errorMessage.toString());
            }

            
            for (int i = 0; i < networkScores.length; i++)
                selectedNetworks[i].setScore(networkScores[i]);
            

            List<ScoredNetwork<E>> newGeneration = evolutionManager.createNewGeneration(
                neuralNetworks.values()
                .parallelStream()
                .sorted()
                .collect(Collectors.toList()),
                
                numNetworks
            );

            synchronized(PREV_GEN_LOCK){
                setCurrentGen(newGeneration);
            }
        }
    }

    public void randomizeNetworks(){
        synchronized(CURRENT_GEN_LOCK){
            for (ScoredNetwork<E> scoredNetwork : neuralNetworks.values())
                NeuralNetworkTools.randomizeWeightsAndBiases(scoredNetwork.get());
            
            if(neuralNetworks.size() < numNetworks)
                addNetworks(evolutionManager.createRandomGeneration(numNetworks - neuralNetworks.size()));
        }
    }

    private void setCurrentGen(List<ScoredNetwork<E>> networks){
        synchronized(CURRENT_GEN_LOCK){
            setPrevGen(neuralNetworks);

            neuralNetworks = new HashMap<>();
            for (ScoredNetwork<E> scoredNetwork : networks)
                neuralNetworks.put(idManager.getNextID(), scoredNetwork);
        }
    }

    private void setPrevGen(Map<Integer, ScoredNetwork<E>> networks){
        synchronized(PREV_GEN_LOCK){
            for (int id : previousGeneration.keySet())
                idManager.releaseID(id);
              
            previousGeneration = networks;
        }
    }

    public Map<Integer, ScoredNetwork<E>> getPreviousGeneration(){
        return Collections.unmodifiableMap(previousGeneration);
    }

    public HashMap<Integer, Map<String, Object>> getMetadata(List<Integer> ids){
        HashMap<Integer, Map<String, Object>> metadataPacket = new HashMap<>(ids.size());
        Map<Integer, Map<String, Object>> synchronizedMetadataPacket = Collections.synchronizedMap(metadataPacket);
        ids.parallelStream().forEach(x -> synchronizedMetadataPacket.put(x, neuralNetworks.get(x).get().getMetadata()));
        return metadataPacket;
    }

    public HashMap<Integer, Map<String, Object>> getMetadata(){
        return getMetadata(neuralNetworks.keySet().stream().collect(Collectors.toList()));
    }

    public List<E> getBestNetworks(){
        synchronized(PREV_GEN_LOCK){
            return RequestHandlerUtils.unpackSuppliers(
                previousGeneration.values()
                .parallelStream()
                .sorted()
                .collect(Collectors.toList())
            );
        }
    }

    public List<E> getBestNetworks(int numNetworks) throws IllegalArgumentException, IndexOutOfBoundsException {
        synchronized(PREV_GEN_LOCK){
            int genSize = previousGeneration.size();

            return RequestHandlerUtils.unpackSuppliers(
                previousGeneration.values()
                .parallelStream()
                .sorted()
                .collect(Collectors.toList())
                .subList(genSize - numNetworks, genSize)
            );
        }
    }


    public List<E> getPreviousGeneration(int from, int to) throws ArrayIndexOutOfBoundsException {
        synchronized(PREV_GEN_LOCK){
            return RequestHandlerUtils.unpackSuppliers(
                previousGeneration.values()
                .parallelStream()
                .collect(Collectors.toList())
                .subList(from, to)
            );
        }
    }

    public Map<Integer, ScoredNetwork<E>> getCurrentGeneration(){
        return Collections.unmodifiableMap(neuralNetworks);
    }

    public State getTrainingState(){
        if(dataTrainedNetworks == null) throw new IllegalStateException("Trainer has not been initialized.");

        if(dataTrainedNetworks.isDone())
            return dataTrainedNetworks.isCancelled() ? State.CANCELLED : State.SUCCESS;
        else
            return State.RUNNING;
    }

    public long getTrainingElapsedTime(){
        if(dataTrainedNetworks == null) throw new IllegalStateException("Trainer has not been initialized.");
        return trainingStartTime.until((trainingEndTime != null) ? trainingEndTime : Instant.now(), ChronoUnit.MILLIS);
    }

    public Double getTrainingBestScore(){
        if(dataTrainedNetworks == null) throw new IllegalStateException("Trainer has not been initialized.");
        return datasetTrainer.getPreviousGeneration()
            .parallelStream()
            .filter(x -> x.getScore().isPresent())
            .mapToDouble(x -> x.getScore().getAsDouble())
            .min()
            .orElse(Double.NaN);
    }

    public int getTrainingSampleCount(){
        return numTrainingSamples;
    }

    public int getTrainingGeneration(){
        if(dataTrainedNetworks == null) throw new IllegalStateException("Trainer has not been initialized.");
        return datasetTrainer.getGeneration();
    }

    public List<RequestHandler> getRequestHandlers(){
        return Arrays.asList(
            new SetupRequest<>(this),
            new RandomizeNetworksRequest<>(this),
            new ProcessInputsRequest<>(this),
            new CreateNewGenerationRequest<>(this),
            new GetBestNetworksRequest<>(this),
            new TrainOnDatasetRequest<>(this),
            new StopTrainingRequest<>(this),
            new GetTrainingStateRequest<>(this)
        );
    }
    
    protected static class TrainingDataSet {
        double[][] inputs;
        double[][] outputs;
    }
    
}

package com.github.thecybrix.simpleneuralnetwork.api.defaults.evolution;

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
import java.util.stream.Collectors;

import com.github.thecybrix.simpleneuralnetwork.api.APIContext;
import com.github.thecybrix.simpleneuralnetwork.api.RequestHandler;
import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils;
import com.github.thecybrix.simpleneuralnetwork.api.RequestHandlerUtils.ParentSelection;
import com.github.thecybrix.simpleneuralnetwork.api.defaults.idmanager.NetworkIDManager;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayout;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkBuilder;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkTools;
import com.github.thecybrix.simpleneuralnetwork.exceptions.DimensionsMismatchException;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.NetworkEvolutionManager;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelector;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.simple.LearningRateDecay;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.simple.SimpleEvolutionManager;

public class EvolutionContext<E extends MutableNeuralNetwork> implements APIContext {
    final private NeuralNetworkBuilder<E> NETWORK_BUILDER;

    final private Object PREV_GEN_LOCK = new Object();
    final private Object CURRENT_GEN_LOCK = new Object();

    final protected NetworkIDManager<? super E> NETWORK_MANAGER;

    protected Map<Integer, ScoredNetwork<E>> previousGeneration = Collections.emptyMap();

    protected NetworkEvolutionManager<E> evolutionManager;
    protected ParentSelector<E> parentSelector;
    
    protected double defaultLearningRate = 2;
    protected double learningRate = defaultLearningRate;
    protected float decayFactor = 0.99f;

    protected int generation = 0;
    protected int numNetworks;
    protected HashMap<Integer, ScoredNetwork<E>> neuralNetworks = new HashMap<>();

    public EvolutionContext(NetworkIDManager<? super E> networkManager, NeuralNetworkBuilder<E> networkBuilder, ParentSelector<E> parentSelector){
        NETWORK_MANAGER = Objects.requireNonNull(networkManager, "Network id manager is null.");
        NETWORK_BUILDER = Objects.requireNonNull(networkBuilder, "Network builder is null.");
        this.parentSelector = (parentSelector != null) ? parentSelector : ParentSelector.eliteSelection();
    }
    

    public void setup(int numNetworks, NetworkLayout layout, ParentSelection parentSelection, List<MutableNeuralNetwork> initialNetworks, boolean createMetadata) throws NoSuchElementException, IllegalArgumentException, DimensionsMismatchException, NullPointerException {

        if(initialNetworks != null)
            NeuralNetworkTools.requireSameDimensions(initialNetworks);
        else if(layout == null)
            throw new IllegalArgumentException("Either layout or initialNetworks must be specified.");

        parentSelector = RequestHandlerUtils.getParentSelector(parentSelection);
        
        NETWORK_BUILDER.reset().withLayout(layout);

        evolutionManager = new SimpleEvolutionManager<>(NETWORK_BUILDER::build, parentSelector);
        evolutionManager.setCreateMetadata(createMetadata);

        this.numNetworks = numNetworks;
        generation = 0;
        learningRate = defaultLearningRate;

        synchronized(PREV_GEN_LOCK){
            if(previousGeneration.size() > 0)
                setPrevGen(Collections.emptyMap());
        }

        synchronized(CURRENT_GEN_LOCK){
            if(initialNetworks != null){
                addNetworks(
                    initialNetworks.parallelStream()
                    .map(x -> new ScoredNetwork<E>(NETWORK_BUILDER.convert(x)))
                    .collect(Collectors.toList())
                );
            }
            
            addNetworks(evolutionManager.createRandomGeneration(numNetworks - neuralNetworks.size()));
        }
        
    }

    public void addNetworks(List<ScoredNetwork<E>> networks){
        for (ScoredNetwork<E> n : networks) 
            neuralNetworks.put(NETWORK_MANAGER.add(n.get()), n);
    }

    public HashMap<Integer, E> getNetworks(List<Integer> ids){
        NetworkIDManager.validateIdCollection(ids, neuralNetworks);
        
        HashMap<Integer, E> networks = new HashMap<>(ids.size());
        for (Integer i : ids) 
            networks.put(i, neuralNetworks.get(i).get());

        return networks;
    }

    public List<ScoredNetwork<E>> getNetworksAsList(List<Integer> ids){
        NetworkIDManager.validateIdCollection(ids, neuralNetworks);
        ArrayList<ScoredNetwork<E>> networks = new ArrayList<>(ids.size());
        for (Integer i : ids) 
            networks.add(neuralNetworks.get(i));
        return networks;
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

                for (int i = 0; i < invalidEntries.size(); i++) {
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
                
                numNetworks,
                learningRate
            );

            generation++;

            learningRate = LearningRateDecay.decay(defaultLearningRate, decayFactor, generation);

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

        learningRate = defaultLearningRate;
        generation = 0;
    }

    protected void setCurrentGen(List<ScoredNetwork<E>> networks){
        synchronized(CURRENT_GEN_LOCK){
            setPrevGen(neuralNetworks);

            neuralNetworks = new HashMap<>();
            addNetworks(networks);
        }
    }

    private void setPrevGen(Map<Integer, ScoredNetwork<E>> networks){
        synchronized(PREV_GEN_LOCK){
            if(!previousGeneration.isEmpty())
                NETWORK_MANAGER.removeAll(previousGeneration.keySet());
              
            previousGeneration = networks;
        }
    }

    public Map<Integer, ScoredNetwork<E>> getPreviousGeneration(){
        return Collections.unmodifiableMap(previousGeneration);
    }

    public HashMap<Integer, Map<String, Object>> getMetadata(List<Integer> ids){
        HashMap<Integer, Map<String, Object>> metadataPacket = new HashMap<>(ids.size());
        ids.stream().forEach(x -> metadataPacket.put(x, neuralNetworks.get(x).get().getMetadata()));
        return metadataPacket;
    }

    public HashMap<Integer, Map<String, Object>> getMetadata(){
        return getMetadata(neuralNetworks.keySet().stream().collect(Collectors.toList()));
    }

    public Map<Integer, E> getNetworks(){
        synchronized(CURRENT_GEN_LOCK){
            HashMap<Integer, E> networks = new HashMap<>(neuralNetworks.size());
            neuralNetworks.entrySet().forEach(x -> networks.put(x.getKey(), x.getValue().get()));
            return networks;
        }
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
    
    public boolean isCreatingMetadata(){
        return evolutionManager.isCreatingMetadata();
    }

    public List<RequestHandler> getRequestHandlers(){
        return Arrays.asList(
            new SetupRequest<>(this),
            new RandomizeNetworksRequest<>(this),
            new CreateNewGenerationRequest<>(this),
            new GetBestNetworksRequest<>(this),
            new GetMetadataRequest<>(this)
        );
    }
}

package com.github.thecybrix.simpleneuralnetwork.api.valuemapping;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import com.github.thecybrix.simpleneuralnetwork.api.evolution.EvolutionContext;
import com.github.thecybrix.simpleneuralnetwork.api.idmanager.NetworkIDManager;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkBuilder;
import com.github.thecybrix.simpleneuralnetwork.server.JsonRequestHandler;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.EvolutionaryTrainer;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelector;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ValueMappingTrainer;

public class ValueMappingContext<E extends MutableNeuralNetwork> extends EvolutionContext<E> {

    public static enum State {
        RUNNING,
        SUCCESS,
        CANCELLED,
        FAILED
    }

    private Future<?> dataTrainedNetworks;
    private EvolutionaryTrainer<E> datasetTrainer;
    private int numTrainingSamples;
    private Instant trainingStartTime, trainingEndTime;

    public ValueMappingContext(NetworkIDManager<? super E> networkManager, NeuralNetworkBuilder<E> networkBuilder, ParentSelector<E> parentSelector){
        super(networkManager, networkBuilder, parentSelector);
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

    public void approximateDataSet(TrainingDataSet dataSet) throws IllegalArgumentException, InterruptedException, ExecutionException, NullPointerException{
        if(datasetTrainer != null) stopTraining();
        
        ValueMappingTrainer<E> trainingScenario = ValueMappingTrainer.of(dataSet.inputs, dataSet.outputs, new ValueMappingTrainer.MeanSquaredError(), NETWORK_MANAGER.FORK_JOIN_POOL);
        datasetTrainer = new EvolutionaryTrainer<>(numNetworks, evolutionManager, trainingScenario);
        datasetTrainer.attachCallback(trainer -> {
            setCurrentGen(trainer.getNetworks());
        });
        datasetTrainer.addAllScored(neuralNetworks.values());
        numTrainingSamples = dataSet.inputs.length;
        trainingEndTime = null;
        dataTrainedNetworks = NETWORK_MANAGER.FORK_JOIN_POOL.submit(() -> {
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

    
    protected static class TrainingDataSet {
        double[][] inputs;
        double[][] outputs;
    }


    public List<JsonRequestHandler> getRequestHandlers(){
        return Arrays.asList(
            new TrainOnDatasetRequest<>(this),
            new StopTrainingRequest<>(this),
            new GetTrainingStateRequest<>(this)
        );
    }

}

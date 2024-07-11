package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.function.BiFunction;

import com.mjsd.simpleneuralnetwork.LossFunctions;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.ScoredNetwork;

public class ValueMappingTrainer<E extends MutableNeuralNetwork, T extends Comparable<T>> implements TrainingScenario<E, T> {
    final private ExecutorService EXECUTOR_SERVICE;
    final private ArrayList<ScoredNetwork<E, T>> NETWORKS = new ArrayList<>();
    final private ArrayList<Future<?>> TASKS = new ArrayList<>();
    private double[][][] predictions;
    final private double[][] INPUTS, OUTPUTS;
    final private BiFunction<double[][], double[][], T> LOSS_FUNCTION;

    
    public ValueMappingTrainer(List<double[]> inputs, List<double[]> outputs, BiFunction<double[][], double[][], T> lossFunction) throws IllegalArgumentException {
        this(inputs.toArray(double[][]::new), outputs.toArray(double[][]::new), lossFunction);
    }
    
    public ValueMappingTrainer(double[][] inputs, double[][] outputs, BiFunction<double[][], double[][], T> lossFunction) throws IllegalArgumentException {
        this(inputs, outputs, lossFunction, Executors.newWorkStealingPool());
    }

    public ValueMappingTrainer(List<double[]> inputs, List<double[]> outputs, BiFunction<double[][], double[][], T> lossFunction, ExecutorService executorService) throws IllegalArgumentException {
        this(inputs.toArray(double[][]::new), outputs.toArray(double[][]::new), lossFunction, executorService);
    }

    public ValueMappingTrainer(double[][] inputs, double[][] outputs, BiFunction<double[][], double[][], T> lossFunction, ExecutorService executorService) throws IllegalArgumentException {
        this.EXECUTOR_SERVICE = Objects.requireNonNull(executorService, "ExecutorService is null.");
        this.LOSS_FUNCTION = Objects.requireNonNull(lossFunction, "LossFunction is null.");

        if(inputs.length != outputs.length)
            throw new IllegalArgumentException("Input and output arrays have differing sizes. (" + inputs.length + " != " + outputs.length + ")");
        if(inputs.length == 0)
            throw new IllegalArgumentException("Input and output lists/arrays are empty.");

        INPUTS = inputs;
        OUTPUTS = outputs;
    }

    
    @Override
    public void run() {
        predictions = new double[NETWORKS.size()][INPUTS.length][];
        TASKS.clear();

        for(int networkIndex = 0; networkIndex < NETWORKS.size(); networkIndex++){
            final MutableNeuralNetwork NETWORK = NETWORKS.get(networkIndex).get();
            final double[][] PREDICTIONS = predictions[networkIndex];
            try {
                TASKS.add(EXECUTOR_SERVICE.submit( () -> mapValues(NETWORK, INPUTS, PREDICTIONS) ));
            } catch (RejectedExecutionException e) {
                e.printStackTrace();
            }
        }

        for (Future<?> task : TASKS)
            try {
                task.get();
            } catch (ExecutionException e) {
                e.printStackTrace();
            } catch (CancellationException | InterruptedException e) {
                continue;
            }
    }


    protected void mapValues(SimpleNeuralNetwork network, double[][] inputs, double[][] outputs){
        // SimpleNeuralNetwork cannot have two seperate threads processing data simultaneously; Thus single threaded.
        for(int groupIndex = 0; groupIndex < inputs.length; groupIndex++){
            network.setInputs(inputs[groupIndex]);
            network.forwardPass();
            outputs[groupIndex] = network.getOutputs();
        }
    }

    @Override
    public void setNetworks(Collection<ScoredNetwork<E, T>> c) {
        NETWORKS.clear();
        NETWORKS.addAll(c);
    }

    @Override
    public void evaluateNetworks() {
        TASKS.clear();
        for(int networkIndex = 0; networkIndex < predictions.length; networkIndex++){
            final int INDEX = networkIndex;
            TASKS.add(EXECUTOR_SERVICE.submit(() -> evaluateNetwork(INDEX)));
        }

        for (Future<?> task : TASKS)
            try {
                task.get();
            } catch (ExecutionException e) {
                e.printStackTrace();
            } catch (CancellationException | InterruptedException e) {
                continue;
            }
    }

    private void evaluateNetwork(int index){
        ScoredNetwork<E, T> network = NETWORKS.get(index);
        T error = LOSS_FUNCTION.apply(predictions[index], OUTPUTS);
        network.setScore(error);
    }

    public static class MeanSquaredError implements BiFunction<double[][], double[][], Double>{

        @Override
        public Double apply(double[][] predictions, double[][] answers) {
            double error = 0;
            for(int i = 0; i < predictions.length; i++)
                error += LossFunctions.meanSquaredError(predictions[i], answers[i]);
            
            return Double.valueOf(error);
        }

    }
    
}

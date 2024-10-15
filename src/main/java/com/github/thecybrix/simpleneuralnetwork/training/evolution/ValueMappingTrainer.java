package com.github.thecybrix.simpleneuralnetwork.training.evolution;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalDouble;
import java.util.Map.Entry;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.function.BiFunction;

import com.github.thecybrix.simpleneuralnetwork.core.LossFunctions;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;

public class ValueMappingTrainer<E extends MutableNeuralNetwork> implements TrainingScenario<E> {
    final private ExecutorService EXECUTOR_SERVICE;
    final private double[][] INPUTS, OUTPUTS;
    final private BiFunction<double[][], double[][], OptionalDouble> LOSS_FUNCTION;
    
    final private ArrayList<Future<?>> TASKS = new ArrayList<>();

    private ValueMappingTrainer(double[][] inputs, double[][] outputs, BiFunction<double[][], double[][], OptionalDouble> lossFunction, ExecutorService executorService) throws IllegalArgumentException, NullPointerException {
        this.EXECUTOR_SERVICE = Objects.requireNonNull(executorService, "ExecutorService is null.");
        this.LOSS_FUNCTION = Objects.requireNonNull(lossFunction, "LossFunction is null.");

        if(inputs.length != outputs.length)
            throw new IllegalArgumentException("Input and output arrays have differing sizes. (" + inputs.length + " != " + outputs.length + ")");
        if(inputs.length == 0)
            throw new IllegalArgumentException("Input and output lists/arrays are empty.");

        INPUTS = inputs;
        OUTPUTS = outputs;
    }
    
    public static <E extends MutableNeuralNetwork> ValueMappingTrainer<E> of(List<double[]> inputs, List<double[]> outputs, BiFunction<double[][], double[][], OptionalDouble> lossFunction) throws IllegalArgumentException, NullPointerException {
        return of(inputs, outputs, lossFunction);
    }

    public static <E extends MutableNeuralNetwork> ValueMappingTrainer<E> of(List<double[]> inputs, List<double[]> outputs, BiFunction<double[][], double[][], OptionalDouble> lossFunction, ExecutorService executorService) throws IllegalArgumentException, NullPointerException {
        return new ValueMappingTrainer<>(inputs.toArray(double[][]::new), outputs.toArray(double[][]::new), lossFunction, executorService);
    }
    
    public static <E extends MutableNeuralNetwork> ValueMappingTrainer<E> of(double[][] inputs, double[][] outputs, BiFunction<double[][], double[][], OptionalDouble> lossFunction) throws IllegalArgumentException, NullPointerException {
        return of(inputs, outputs, lossFunction, Executors.newWorkStealingPool());
    }
    
    public static <E extends MutableNeuralNetwork> ValueMappingTrainer<E> of(double[][] inputs, double[][] outputs, BiFunction<double[][], double[][], OptionalDouble> lossFunction, ExecutorService executorService) throws IllegalArgumentException, NullPointerException {
        return new ValueMappingTrainer<>(inputs, outputs, lossFunction, executorService);
    }

    public static <E extends MutableNeuralNetwork> ValueMappingTrainer<E> of(Map<double[], double[]> dataSet, BiFunction<double[][], double[][], OptionalDouble> lossFunction) throws IllegalArgumentException, NullPointerException{
        return of(dataSet, lossFunction, Executors.newWorkStealingPool());
    }

    public static <E extends MutableNeuralNetwork> ValueMappingTrainer<E> of(Map<double[], double[]> dataSet, BiFunction<double[][], double[][], OptionalDouble> lossFunction, ExecutorService executorService) throws IllegalArgumentException, NullPointerException{
        double[][] inputs = new double[dataSet.size()][];
        double[][] outputs = new double[dataSet.size()][];
        int index = 0;
        for (Entry<double[], double[]> pair : dataSet.entrySet()) {
            inputs[index] = pair.getKey();
            outputs[index] = pair.getValue();
            index++;
        }
        return new ValueMappingTrainer<>(inputs, outputs, lossFunction, executorService);
    }

    
    @Override
    public void execute(Collection<ScoredNetwork<E>> networks) {

        for(ScoredNetwork<E> n : networks){
            try {
                TASKS.add(EXECUTOR_SERVICE.submit( () -> evaluateNetwork(n) ));
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

        TASKS.clear();
    }


    protected void evaluateNetwork(ScoredNetwork<E> scoredNetwork){
        double[][] predictions = new double[INPUTS.length][];
        
        // SimpleNeuralNetwork cannot have two seperate threads processing data simultaneously; Thus single threaded.
        E network = scoredNetwork.get();

        synchronized(network){
            for(int i = 0; i < INPUTS.length; i++){
                network.setInputs(INPUTS[i]);
                network.forwardPass();
                predictions[i] = network.getOutputs();
            }

            OptionalDouble error = LOSS_FUNCTION.apply(predictions, OUTPUTS);
            scoredNetwork.setScore(error);
        }
    }


    public static class MeanSquaredError implements BiFunction<double[][], double[][], OptionalDouble>{

        @Override
        public OptionalDouble apply(double[][] predictions, double[][] trueValues) {
            double error = 0;
            for(int i = 0; i < predictions.length; i++)
                error += LossFunctions.meanSquaredError(predictions[i], trueValues[i]);
            
            return OptionalDouble.of(error);
        }

    }
    
}

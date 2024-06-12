package com.mjsd.simpleneuralnetwork;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.function.Function;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import com.mjsd.simpleneuralnetwork.Serialization.NetworkDeserializer;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.ActivationFunction;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.InputNormalizer;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.InputProvider;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.OutputHandler;
import com.mjsd.simpleneuralnetwork.exceptions.DimensionsMismatchException;
import com.mjsd.simpleneuralnetwork.gson.CustomGsonFactory;

public class NeuralNetworkBuilder<E extends SimpleNeuralNetwork> {

    final private Function<NetworkLayout, E> NETWORK_SUPPLIER;
    private NetworkLayoutBuilder layoutBuilder;
    private List<InputProvider> inputProviders;
    private List<OutputHandler> outputHandlers;
    private Optional<double[][][]> weights = Optional.empty();
    private Optional<double[][]> biases = Optional.empty();
    private Optional<ExecutorService> inputFetchingService = Optional.empty(),
                              outputHandlingService = Optional.empty(),
                              forwardPassService = Optional.empty();

    public NeuralNetworkBuilder(Function<NetworkLayout, E> supplier) throws NullPointerException{
        NETWORK_SUPPLIER = Objects.requireNonNull(supplier);
        layoutBuilder = new NetworkLayoutBuilder();
        inputProviders = null;
        outputHandlers = null;
    }

    public NeuralNetworkBuilder(Function<NetworkLayout, E> supplier, E initialState) throws NullPointerException{
        NETWORK_SUPPLIER = Objects.requireNonNull(supplier);
        Objects.requireNonNull(initialState);
        this.layoutBuilder = new NetworkLayoutBuilder(initialState.getLayout());
        inputProviders = initialState.getInputProviders();
        outputHandlers = initialState.getOutputHandlers();

        this.weights = Optional.of(NeuralNetworkTools.deepCopy(initialState.getWeights()));
        this.biases = Optional.of(NeuralNetworkTools.deepCopy(initialState.getBiases()));
    }

    public NeuralNetworkBuilder(Function<NetworkLayout, E> supplier, NetworkLayout initialState) throws NullPointerException{
        this(supplier, initialState, null, null);
    }

    public NeuralNetworkBuilder(Function<NetworkLayout, E> supplier, NetworkLayout initialState, List<InputProvider> inputProviders, List<OutputHandler> outputHandlers) throws NullPointerException{
        NETWORK_SUPPLIER = Objects.requireNonNull(supplier);
        this.layoutBuilder = new NetworkLayoutBuilder(initialState);
        withInputProviders(inputProviders);
        withOutputHandlers(outputHandlers);
    }


    /**
     * Creates a NeuralNetwork based on the state of this builder.
     * @return A new NeuralNetwork object.
     * @throws IllegalStateException If {@code withInputLayer()} or {@code withOutputLayer()} are not called prior to {@code create()}.
     * @see #withInputLayer()
     * @see #withOutputLayer()
     */
    public E build() throws IllegalStateException {
        E instance = NETWORK_SUPPLIER.apply(layoutBuilder.build());
        instance.inputProtocol = inputFetchingService.isPresent() ? new ParallelInputFetcher(instance, inputFetchingService.get()) : instance::getInputsLinear;
        instance.outputProtocol = outputHandlingService.isPresent() ? new ParallelOutputHandler(instance, outputHandlingService.get()) : instance::handleOutputsLinear;
        instance.forwardPassProtocol = forwardPassService.isPresent() ? new ParallelForwardPass(instance, forwardPassService.get()) : instance::forwardPassLinear;

        try {
            if(inputProviders != null) instance.setInputProviders(inputProviders);
            if(outputHandlers != null) instance.setOutputHandlers(outputHandlers);

            if(biases.isPresent()) NeuralNetworkTools.ensureValidBiasArray(instance.LAYOUT, biases.get());
            if(weights.isPresent()) NeuralNetworkTools.ensureValidWeightArray(instance.LAYOUT, weights.get());

            if(biases.isPresent()) instance.biases = NeuralNetworkTools.deepCopy(biases.get());
            if(weights.isPresent()) instance.weights = NeuralNetworkTools.deepCopy(weights.get());

        } catch (IllegalArgumentException | DimensionsMismatchException e) {
            throw new IllegalStateException("Unable to build neural network from current state.", e);
        }

        return instance;
    }

    public void reset(){
        layoutBuilder.reset();
        inputProviders = null;
        outputHandlers = null;
        weights = null;
        biases = null;
    }

    public NeuralNetworkBuilder<E> withLayout(NetworkLayout layout) throws NullPointerException{
        layoutBuilder.setState(layout);
        return this;
    }

    public NeuralNetworkBuilder<E> withInputLayer(int size){
        layoutBuilder.withInputLayer(size);
        return this;
    }

    public NeuralNetworkBuilder<E> withInputLayer(int size, ActivationFunction activationFunction){
        layoutBuilder.withInputLayer(size, activationFunction);
        return this;
    }

    public NeuralNetworkBuilder<E> withInputLayer(int size, InputNormalizer inputNormalizer){
        layoutBuilder.withInputLayer(size, inputNormalizer);
        return this;
    }

    public NeuralNetworkBuilder<E> withInputLayer(int size, InputNormalizer inputNormalizer, ActivationFunction activationFunction){
        layoutBuilder.withInputLayer(size, inputNormalizer, activationFunction);
        return this;
    }

    public NeuralNetworkBuilder<E> withOutputLayer(int size){
        layoutBuilder.withOutputLayer(size);
        return this;
    }

    public NeuralNetworkBuilder<E> withOutputLayer(int size, InputNormalizer inputNormalizer){
        layoutBuilder.withOutputLayer(size, inputNormalizer);
        return this;
    }

    public NeuralNetworkBuilder<E> withOutputLayer(int size, ActivationFunction activationFunction){
        layoutBuilder.withOutputLayer(size, activationFunction);
        return this;
    }

    public NeuralNetworkBuilder<E> withOutputLayer(int size, InputNormalizer inputNormalizer, ActivationFunction activationFunction){
        layoutBuilder.withOutputLayer(size, inputNormalizer, activationFunction);
        return this;
    }

    public NeuralNetworkBuilder<E> addHiddenLayer(int size) throws IllegalArgumentException{
        layoutBuilder.addLayer(size);
        return this;
    }

    public NeuralNetworkBuilder<E> addHiddenLayer(int size, InputNormalizer inputNormalizer) throws IllegalArgumentException{
        layoutBuilder.addLayer(size, inputNormalizer);
        return this;
    }

    public NeuralNetworkBuilder<E> addHiddenLayer(int size, ActivationFunction activationFunction) throws IllegalArgumentException{
        layoutBuilder.addLayer(size, activationFunction);
        return this;
    }

    public NeuralNetworkBuilder<E> addHiddenLayer(int size, InputNormalizer inputNormalizer, ActivationFunction activationFunction) throws IllegalArgumentException{
        layoutBuilder.addLayer(size, inputNormalizer, activationFunction);
        return this;
    }

    public NeuralNetworkBuilder<E> addHiddenLayers(int numLayers, int size) throws IllegalArgumentException {
        layoutBuilder.addLayers(numLayers, size);
        return this;
    }

    public NeuralNetworkBuilder<E> addHiddenLayers(int numLayers, int size, InputNormalizer inputNormalizer) throws IllegalArgumentException{
        layoutBuilder.addLayers(numLayers, size, inputNormalizer);
        return this;
    }

    public NeuralNetworkBuilder<E> addHiddenLayers(int numLayers, int size, ActivationFunction activationFunction) throws IllegalArgumentException{
        layoutBuilder.addLayers(numLayers, size, activationFunction);
        return this;
    }

    public NeuralNetworkBuilder<E> addHiddenLayers(int numLayers, int size, InputNormalizer inputNormalizer, ActivationFunction activationFunction) throws IllegalArgumentException{
        layoutBuilder.addLayers(numLayers, size, inputNormalizer, activationFunction);
        return this;
    }

    public NeuralNetworkBuilder<E> withBiases(double[][] biases) {
        this.biases = Optional.ofNullable(biases);
        return this;
    }

    public NeuralNetworkBuilder<E> withWeights(double[][][] weights) {
        this.weights = Optional.ofNullable(weights);
        return this;
    }
    
    public NeuralNetworkBuilder<E> withInputProviders(List<InputProvider> inputProviders) {
        this.inputProviders = (inputProviders == null) ? null : new ArrayList<>(inputProviders);
        return this;
    }
    
    public NeuralNetworkBuilder<E> withoutInputProviders() {
        this.inputProviders = null;
        return this;
    }

    public NeuralNetworkBuilder<E> withOutputHandlers(List<OutputHandler> outputHandlers) {
        this.outputHandlers = (outputHandlers == null) ? null : new ArrayList<>(outputHandlers);
        return this;
    }

    public NeuralNetworkBuilder<E> withoutOutputHandlers() {
        this.outputHandlers = null;
        return this;
    }

    public NeuralNetworkBuilder<E> enableParallelInputFetching(ExecutorService executorService){
        this.inputFetchingService = Optional.ofNullable(executorService);
        return this;
    }

    public NeuralNetworkBuilder<E> disableParallelInputFetching(){
        this.inputFetchingService = Optional.empty();
        return this;
    }

    public NeuralNetworkBuilder<E> enableParallelOutputHandling(ExecutorService executorService){
        this.outputHandlingService = Optional.ofNullable(executorService);
        return this;
    }

    public NeuralNetworkBuilder<E> disableParallelOutputHandling(){
        this.outputHandlingService = Optional.empty();
        return this;
    }
    
    public NeuralNetworkBuilder<E> enableParallelForwardPass(ExecutorService executorService){
        this.forwardPassService = Optional.ofNullable(executorService);
        return this;
    }

    public NeuralNetworkBuilder<E> disableParallelForwardPass(){
        this.forwardPassService = Optional.empty();
        return this;
    }

    public NetworkLayout getLayout() throws IllegalStateException {
        return layoutBuilder.build();
    }

    public static <T extends SimpleNeuralNetwork> T fromJson(String json, Class<T> classOfT) throws JsonSyntaxException{
        Gson gson = CustomGsonFactory.getInstance();
        return gson.fromJson(json, classOfT);
    }

    public static <T extends SimpleNeuralNetwork> NeuralNetworkBuilder<T> loadBinary(String filePath, NeuralNetworkBuilder<T> builder) throws FileNotFoundException, IOException, SecurityException, NullPointerException{
        NetworkDeserializer deserializer = new NetworkDeserializer();
        return deserializer.load(filePath, builder);
    }


	private class ParallelForwardPass implements Runnable{
		final private ArrayList<Future<Double>> NODE_VALUES;
        final private SimpleNeuralNetwork PARENT;
        final private ExecutorService EXECUTOR_SERVICE;

		public ParallelForwardPass(SimpleNeuralNetwork parent, ExecutorService executorService){
            PARENT = Objects.requireNonNull(parent, "Network is null.");
            EXECUTOR_SERVICE = Objects.requireNonNull(executorService, "ExecutorService is null.");
			int initialSize = PARENT.getLayout().getHiddenLayers().stream()
									.map(x -> x.getNodeCount())
									.max((x, y) -> x.compareTo(y))
									.get();
			NODE_VALUES = new ArrayList<Future<Double>>(initialSize);
		}

		@Override
		public void run() {
			SimpleNeuralNetwork.applyLayerModifiers(PARENT.inputs, PARENT.inputNormalizer, PARENT.inputActivation);

			double[] previousLayer = PARENT.inputs;

			for(int layer = 0; layer < PARENT.hiddenLayers.length; layer++){
                dotSequence(previousLayer, PARENT.weights[layer], PARENT.hiddenLayers[layer]);
				NeuralNetworkTools.vectorSum(PARENT.hiddenLayers[layer], PARENT.biases[layer], PARENT.hiddenLayers[layer]);
				SimpleNeuralNetwork.applyLayerModifiers(PARENT.hiddenLayers[layer], PARENT.hiddenNormalizers[layer], PARENT.hiddenActivations[layer]);
				previousLayer = PARENT.hiddenLayers[layer];
			}

			dotSequence(previousLayer, PARENT.weights[PARENT.OUTPUT_LAYER], PARENT.outputs);
			NeuralNetworkTools.vectorSum(PARENT.outputs, PARENT.biases[PARENT.OUTPUT_LAYER], PARENT.outputs);
			SimpleNeuralNetwork.applyLayerModifiers(PARENT.outputs, PARENT.outputNormalizer, PARENT.outputActivation);
		}

        private void dotSequence(double[] vector1, double[][] crMatrix, double[] result){
            NODE_VALUES.clear();

            for (int i = 0; i < crMatrix.length; i++){
                NODE_VALUES.add(EXECUTOR_SERVICE.submit(new DotProduct(vector1, crMatrix[i])));
            }

            for(int i = 0; i < NODE_VALUES.size(); i++)
                try {
                    result[i] = NODE_VALUES.get(i).get();
                } catch (ExecutionException | CancellationException e) {
                    continue;
                } catch (InterruptedException e){
                    break;
                }
        }

        private class DotProduct implements Callable<Double>{
            final double[] VECTOR_1, VECTOR_2;

            public DotProduct(double[] v1, double[] v2){
                VECTOR_1 = v1;
                VECTOR_2 = v2;
            }

            @Override
            public Double call() {
                return NeuralNetworkTools.dotProduct(VECTOR_1, VECTOR_2);
            }

        } 

	}

	private class ParallelInputFetcher implements Runnable {
        final private SimpleNeuralNetwork PARENT;
		private List<Callable<?>> inputOperations;
        final private ExecutorService EXECUTOR_SERVICE;

        public ParallelInputFetcher(SimpleNeuralNetwork parent, ExecutorService executorService){
            PARENT = Objects.requireNonNull(parent, "Network is null.");
            EXECUTOR_SERVICE = Objects.requireNonNull(executorService, "ExecutorService is null.");
            inputOperations = Arrays.asList(new Callable[PARENT.inputProviders.length]);

            for(int i = 0; i < PARENT.inputProviders.length; i++) {
				final int INDEX = i;
				inputOperations.set(i, () -> { PARENT.inputs[INDEX] = PARENT.inputProviders[INDEX].orElse(PARENT.inputs[INDEX]); return null; });	
			}
		}

		@Override
		public void run() {
			try {
				EXECUTOR_SERVICE.invokeAll(inputOperations);
			} catch (InterruptedException | RejectedExecutionException e) {}
		}

	}

	private class ParallelOutputHandler implements Runnable{
        final private SimpleNeuralNetwork PARENT;
		private List<Callable<?>> outputOperations;
        final private ExecutorService EXECUTOR_SERVICE;

        public ParallelOutputHandler(SimpleNeuralNetwork parent, ExecutorService executorService){        
            PARENT = Objects.requireNonNull(parent, "Network is null.");
            EXECUTOR_SERVICE = Objects.requireNonNull(executorService, "ExecutorService is null.");
            outputOperations = Arrays.asList(new Callable[PARENT.outputHandlers.length]);

			for (int i = 0; i < PARENT.outputHandlers.length; i++) {
                final int INDEX = i;
				outputOperations.set(i, () -> { PARENT.outputHandlers[INDEX].handle(PARENT.outputs[INDEX]); return null; });	
			}
        }

		@Override
		public void run(){
			try {
				EXECUTOR_SERVICE.invokeAll(outputOperations);
			} catch (InterruptedException | RejectedExecutionException e) {}
		}
	}

}

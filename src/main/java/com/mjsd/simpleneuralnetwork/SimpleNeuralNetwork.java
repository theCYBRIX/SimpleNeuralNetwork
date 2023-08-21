
package com.mjsd.simpleneuralnetwork;

import com.google.gson.Gson;
import com.google.gson.annotations.JsonAdapter;

import com.mjsd.simpleneuralnetwork.NetworkLayout.NetworkLayer;
import com.mjsd.simpleneuralnetwork.exceptions.DimensionsMismatchException;
import com.mjsd.simpleneuralnetwork.gson.*;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

@JsonAdapter(SimpleNeuralNetworkAdapter.class)
public class SimpleNeuralNetwork {

	final protected int OUTPUT_LAYER;
	final protected NetworkLayout LAYOUT;

	protected double[][][] weights;
	protected double[][] biases;

	protected double[][] hiddenLayers;
	protected ActivationFunction[] hiddenActivations;
	protected InputNormalizer[] hiddenNormalizers;

	protected double[] inputs;
	protected ActivationFunction inputActivation;
	protected InputNormalizer inputNormalizer;

	protected double[] outputs;
	protected ActivationFunction outputActivation;
	protected InputNormalizer outputNormalizer;

	private OutputHandler[] outputHandlers;
	private InputProvider[] inputProviders;

	final private Object SYNCH_OBJECT = new Object();
	private boolean parallel = false;
	private Runnable inputProtocol = this::getInputsLinear,
					 outputProtocol = this::handleOutputsLinear;

	protected SimpleNeuralNetwork(SimpleNeuralNetwork template) throws NullPointerException{
		this(template.LAYOUT, NeuralNetworkTools.deepCopy(template.weights), NeuralNetworkTools.deepCopy(template.biases), Arrays.copyOf(template.outputHandlers, template.outputHandlers.length), Arrays.copyOf(template.inputProviders, template.inputProviders.length));
	}

	
	protected SimpleNeuralNetwork(NetworkLayout layout) throws NullPointerException {
		LAYOUT = Objects.requireNonNull(layout);

		NetworkLayer inputLayerLayout = layout.getInputLayer(),
					 outputLayerLayout = layout.getOutputLayer();
		List<NetworkLayer> hiddenLayerLayouts = layout.getHiddenLayers();

		this.outputHandlers = new OutputHandler[outputLayerLayout.getNodeCount()];
		for (int i = 0; i < outputHandlers.length; i++)
			outputHandlers[i] = OutputHandler.NO_HANDLER;
			
		this.inputProviders = new InputProvider[inputLayerLayout.getNodeCount()];
		for (int i = 0; i < inputProviders.length; i++)
			inputProviders[i] = InputProvider.NO_PROVIDER;


		this.hiddenLayers = new double[hiddenLayerLayouts.size()][];
		this.hiddenActivations = new ActivationFunction[hiddenLayerLayouts.size()];
		this.hiddenNormalizers = new InputNormalizer[hiddenLayerLayouts.size()];
		for (int i = 0; i < hiddenLayerLayouts.size(); i++) {
			NetworkLayer layerLayout = hiddenLayerLayouts.get(i);
			this.hiddenLayers[i] = new double[layerLayout.getNodeCount()];
			this.hiddenActivations[i] = layerLayout.getActivationFunction();
			this.hiddenNormalizers[i] = layerLayout.getInputNormalizer();
		}

		this.inputs= new double[inputLayerLayout.getNodeCount()];
		this.inputActivation = inputLayerLayout.getActivationFunction();
		this.inputNormalizer = inputLayerLayout.getInputNormalizer();

		this.outputs = new double[outputLayerLayout.getNodeCount()];
		this.outputActivation = outputLayerLayout.getActivationFunction();
		this.outputNormalizer = outputLayerLayout.getInputNormalizer();

		this.weights = new double[hiddenLayers.length + 1][][];
		this.biases = new double[weights.length][];

		double[] previousLayer = inputs;
		for (int layer = 0; layer < hiddenLayers.length; layer++) {
			this.weights[layer] = new double[hiddenLayers[layer].length][];
			this.biases[layer] = new double[hiddenLayers[layer].length];
			for (int node = 0; node < weights[layer].length; node++)
				this.weights[layer][node] = new double[previousLayer.length];
			previousLayer = hiddenLayers[layer];
		}

		OUTPUT_LAYER = hiddenLayers.length;

		this.weights[OUTPUT_LAYER] = new double[outputs.length][];
		this.biases[OUTPUT_LAYER] = new double[outputs.length];
		for (int node = 0; node < outputs.length; node++)
			this.weights[OUTPUT_LAYER][node] = new double[previousLayer.length];

	}

	
	/**
	 * @implNote This constructor does not check the given values, and should only be used if the inputs are guaranteed to be valid.
	 */
	protected SimpleNeuralNetwork(NetworkLayout layout, double[][][] weights, double[][] biases, OutputHandler[] outputHandlers, InputProvider[] inputProviders) throws NullPointerException {
		LAYOUT = layout;
		this.weights = weights;
		this.biases = biases;

		NetworkLayer inputLayerLayout = layout.getInputLayer(),
					 outputLayerLayout = layout.getOutputLayer();
		List<NetworkLayer> hiddenLayerLayouts = layout.getHiddenLayers();

		this.outputHandlers = outputHandlers;
		this.inputProviders = inputProviders;

		this.hiddenLayers = new double[hiddenLayerLayouts.size()][];
		this.hiddenActivations = new ActivationFunction[hiddenLayerLayouts.size()];
		this.hiddenNormalizers = new InputNormalizer[hiddenLayerLayouts.size()];
		for (int i = 0; i < hiddenLayerLayouts.size(); i++) {
			NetworkLayer layerLayout = hiddenLayerLayouts.get(i);
			this.hiddenLayers[i] = new double[layerLayout.getNodeCount()];
			this.hiddenActivations[i] = layerLayout.getActivationFunction();
			this.hiddenNormalizers[i] = layerLayout.getInputNormalizer();
		}

		this.inputs= new double[inputLayerLayout.getNodeCount()];
		this.inputActivation = inputLayerLayout.getActivationFunction();
		this.inputNormalizer = inputLayerLayout.getInputNormalizer();

		this.outputs = new double[outputLayerLayout.getNodeCount()];
		this.outputActivation = outputLayerLayout.getActivationFunction();
		this.outputNormalizer = outputLayerLayout.getInputNormalizer();

		OUTPUT_LAYER = hiddenLayers.length;
	}

	public void forwardPass(){
		synchronized(SYNCH_OBJECT){
			inputProtocol.run();

			applyLayerModifiers(inputs, inputNormalizer, inputActivation);

			double[] previousLayer = inputs;

			for(int layer = 0; layer < hiddenLayers.length; layer++){
				hiddenLayers[layer] = vectorSum(dotSequence(previousLayer, weights[layer]), biases[layer]);
				applyLayerModifiers(hiddenLayers[layer], hiddenNormalizers[layer], hiddenActivations[layer]);
				previousLayer = hiddenLayers[layer];
			}

			outputs = vectorSum(dotSequence(previousLayer, weights[OUTPUT_LAYER]), biases[OUTPUT_LAYER]);
			applyLayerModifiers(outputs, outputNormalizer, outputActivation);
		
			outputProtocol.run();
		}
	}

	private void getInputsLinear(){
		for (int i = 0; i < inputs.length; i++)
			inputs[i] = inputProviders[i].orElse(inputs[i]);
	}

	private void handleOutputsLinear(){
		for (int i = 0; i < outputHandlers.length; i++)
			outputHandlers[i].handle(outputs[i]);
	}

	private static double dotProduct(double[] v1, double[] v2) {
		double product = 0;

		for (int i = 0; i < v1.length; i++)
			product += v1[i] * v2[i];

		return product;
	}

	private static double[] dotSequence(double[] v1, double[][] crMatrix) {
		double product[] = new double[crMatrix.length];

		for (int column = 0; column < crMatrix.length; column++)
			product[column] = dotProduct(v1, crMatrix[column]);

		return product;
	}

	private static double[] vectorSum(double[] v1, double[] v2) {
		double sum[] = new double[v1.length];

		for (int i = 0; i < v1.length; i++)
			sum[i] = v1[i] + v2[i];

		return sum;
    }
	
	public SimpleNeuralNetwork copy() {
		return new SimpleNeuralNetwork(this);
	}

	/**
	 * @return A NeuralNetworkBuilder initialized to the state of the this network.
	 */
	public NeuralNetworkBuilder<? extends SimpleNeuralNetwork> newBuilder(){
		return new SimpleNeuralNetworkBuilder(this);
	}

	@Override
	public String toString() {
		return SimpleNeuralNetwork.toString(this);
	}

	private static String toString(SimpleNeuralNetwork network){
		StringBuilder out = new StringBuilder();
		out.append("input: ");
		arrayToString(out, network.inputs);
		for(int i = 0, next = 1; i < network.hiddenLayers.length; next++){
			out.append("\nlayer ").append(next).append(": ");
			arrayToString(out, network.hiddenLayers[i]);
			i = next;
		}
		out.append("\noutput: ");
		arrayToString(out, network.outputs);
		return out.toString();
	}


    private static void arrayToString(StringBuilder appendTo, double[] array){
        appendTo.append("[").append(array[0]);
        for(int i = 1; i < array.length; i++)
            appendTo.append(", ").append(array[i]);
        appendTo.append("]");
    }

	@Override
	public boolean equals(Object obj) {
		if(!(obj instanceof SimpleNeuralNetwork)) return false;
		SimpleNeuralNetwork other = (SimpleNeuralNetwork)obj;

		if(!NeuralNetworkTools.haveSameLayout(this, other)) return false;

		double[][][] otherWeights = other.weights;

		for(int layer = 0; layer < weights.length; layer++)
			for(int node = 0; node < weights[layer].length; node++)
				for(int weight = 0; weight < weights[layer][node].length; weight++)
					if(this.weights[layer][node][weight] != otherWeights[layer][node][weight])
						return false;

		double[][] otherBiases = other.biases;

		for(int layer = 0; layer < biases.length; layer++)
			for(int node = 0; node < biases[layer].length; node++)
				if(this.biases[layer][node] != otherBiases[layer][node])
					return false;

		return true;
	}
	






	/*******************************************************************************************************************
	***************************************************** Getters ******************************************************
	*******************************************************************************************************************/

	public double getInput(int index) throws IndexOutOfBoundsException {
		return inputs[index];
	}

	public double[] getInputLayer() {
		return inputs;
	}

	public List<InputProvider> getInputProviders() {
		return new ArrayList<>(Arrays.asList(inputProviders));
	}

	public double getOutput(int index) throws IndexOutOfBoundsException {
		return outputs[index];
	}

	public double[] getOutputLayer() {
		return outputs;
	}

	public List<OutputHandler> getOutputHandlers() {
		return new ArrayList<>(Arrays.asList(outputHandlers));
	}

	public NetworkLayout getLayout(){
		return LAYOUT;
	}

	protected NetworkLayer getHiddenLayerLayout(int layerIndex) throws IndexOutOfBoundsException {
		return LAYOUT.HIDDEN_LAYERS.get(layerIndex);
	}

	public double getValue(int hiddenLayerIndex, int nodeIndex) throws ArrayIndexOutOfBoundsException {
		synchronized(SYNCH_OBJECT){
			return hiddenLayers[hiddenLayerIndex][nodeIndex];
		}
	}

    public String toJson(){
        return this.toJson(CustomGsonFactory.getInstance());
    }

    public String toJson(Gson gson){
        return gson.toJson(this);
    }

	/**
	 * Gets the 2D array containing the values of the Neural Network's
	 * nodes in the form of {@code hiddenLayers[layer][node]}.
	 * 
	 * @return An array that functions as a column row matrix.
	 * @apiNote The returned array is not a copy. The neural network will modify it's contents with each call to {@link #forwardPass()}.
	 */
	protected double[][] getHiddenLayers() {
		return hiddenLayers;
	}






	/*******************************************************************************************************************
	***************************************************** Setters ******************************************************
	*******************************************************************************************************************/

	public void setInput(double[] values) throws NullPointerException, DimensionsMismatchException {
		if (Objects.requireNonNull(values).length != this.inputs.length)
			throw new DimensionsMismatchException("Number of values (" + values.length + ") does not match number of input nodes ("
					+ this.inputs.length + ").");

		synchronized(SYNCH_OBJECT){
			for (int i = 0; i < this.inputs.length; i++)
				this.inputs[i] = values[i];
		}
	}

	public void setInput(int index, double value) throws ArrayIndexOutOfBoundsException {
		synchronized(SYNCH_OBJECT){
			inputs[index] = value;
		}
	}

	public void setOutputHandler(int index, OutputHandler outputHandler) throws IndexOutOfBoundsException {
		if(index < 0 || index > outputHandlers.length) throw new IndexOutOfBoundsException(index);
		synchronized (SYNCH_OBJECT) {
			this.outputHandlers[index] = OutputHandler.ensureHandler(outputHandler);
		}
	}

	public void setInputProvider(int index, InputProvider inputProvider) throws IndexOutOfBoundsException {
		if(index < 0 || index > inputProviders.length) throw new IndexOutOfBoundsException(index);
		synchronized (SYNCH_OBJECT) {
			this.inputProviders[index] = InputProvider.ensureProvider(inputProvider);
		}
	}

	public void setOutputHandlers(List<OutputHandler> outputHandlers) throws IllegalArgumentException {
		if(Objects.requireNonNull(outputHandlers, "List is null.").size() != outputs.length)
			throw new IllegalArgumentException("Number of objects in list doesn't match number of output nodes. (" + outputHandlers.size() + " != " + outputs.length + ")");
		synchronized (SYNCH_OBJECT) {
			for (int i = 0; i < outputs.length; i++)
				this.outputHandlers[i] = OutputHandler.ensureHandler(outputHandlers.get(i));
		}
	}

	public void setInputProviders(List<InputProvider> inputProviders) throws IllegalArgumentException {
		if(Objects.requireNonNull(inputProviders, "List is null.").size() != inputs.length)
			throw new IllegalArgumentException("Number of objects in list doesn't match number of input nodes. (" + inputProviders.size() + " != " + inputs.length + ")");
		synchronized (SYNCH_OBJECT) {
			for (int i = 0; i < inputs.length; i++)
				this.inputProviders[i] = InputProvider.ensureProvider(inputProviders.get(i));
		}
	}



	/**
	 * @return a deep copy of the 3-dimensional array containing the weights of the network; consisting of {@code weights[layer][node][weight]}.
	 * @see {@link MutableNeuralNetwork#retrieveWeightsArray() }
	 */
	final public double[][][] getWeights()  {
		synchronized(SYNCH_OBJECT){
			return NeuralNetworkTools.deepCopy(weights);
		}
	}
	
	/**
	 * @return A deep copy of the 2-dimensional array containing the biases of this network; consisting of {@code biases[layer][node]}.
	 * @see {@link MutableNeuralNetwork#retrieveBiasesArray() }
	 */
	final public double[][] getBiases(){
		synchronized(SYNCH_OBJECT){
			return NeuralNetworkTools.deepCopy(biases);
		}
	}


	public void setParallel(boolean enabled) {
		if(parallel == enabled) return;
		parallel = enabled;
		if(parallel){
			ParallelOperationAdapter adapter = new ParallelOperationAdapter();
			inputProtocol = adapter::getInputs;
			outputProtocol = adapter::handleOutputs;
		} else {
			inputProtocol = this::getInputsLinear;
			outputProtocol = this::handleOutputsLinear;
		}
	}



	/*******************************************************************************************************************
	************************************************* Static Methods ***************************************************
	*******************************************************************************************************************/

	private static void applyLayerModifiers(double[] layer, InputNormalizer normalizer, ActivationFunction activationFunction){
		normalizer.normalize(layer);

		for(int i = 0; i < layer.length; i++)
			layer[i] = activationFunction.apply(layer, i);
	}

	@JsonAdapter(ActivationFunctionAdapter.class)
	@FunctionalInterface
	public interface ActivationFunction{
		public double apply(double[] layer, int index);

        public static ActivationFunction ensureFunction(ActivationFunction function){
            return (function == null) ? ActivationFunctions.LINEAR : function;
        }
	}

	@JsonAdapter(InputNormalizerAdapter.class)
	@FunctionalInterface
	public interface InputNormalizer {
		public void normalize(double[] values);

        public static InputNormalizer ensureNormalizer(InputNormalizer normalizer){
            return (normalizer == null) ? InputNormalizers.NO_NORMALIZER : normalizer;
        }
	}

	@FunctionalInterface
	public interface OutputHandler {
		final static OutputHandler NO_HANDLER = x -> {};

		public void handle(double output);

		public static OutputHandler ensureHandler(OutputHandler handler){
			return (handler == null) ? OutputHandler.NO_HANDLER : handler;
		}

		public static OutputHandler none(){
			return OutputHandler.NO_HANDLER;
		}
	}

	@FunctionalInterface
	public interface InputProvider {
		final static InputProvider NO_PROVIDER = x -> x;

		public double orElse(double value);

		private static InputProvider ensureProvider(InputProvider provider){
			return (provider == null) ? InputProvider.none() : provider;
		}

		private static InputProvider none(){
			return NO_PROVIDER;
		}
	}

	private class ParallelOperationAdapter{
		private ExecutorService executorService = Executors.newCachedThreadPool();

		@SuppressWarnings("unchecked")
		private Future<Double>[] inputOperations = new Future[inputProviders.length];
		private Future<?>[] outputOperations = new Future[outputHandlers.length];

		public void getInputs(){
			int i;
			for (i = 0; i < inputProviders.length; i++) {
				final InputProvider INPUT_PROVIDER = inputProviders[i];
				final double CURRENT_VALUE = inputs[i];
				inputOperations[i] = executorService.submit(() -> INPUT_PROVIDER.orElse(CURRENT_VALUE));	
			}

			for(i = 0; i < inputProviders.length; i++)
				try {
					inputs[i] = inputOperations[i].get();
				} catch (CancellationException | ExecutionException e) {
					continue;
				} catch (InterruptedException e) {
					break;
				}
		}

		public void handleOutputs(){
			for (int i = 0; i < outputHandlers.length; i++) {
				OutputHandler handler = outputHandlers[i];
				double output = outputs[i];
				outputOperations[i] = executorService.submit(() -> handler.handle(output));	
			}
			for (int i = 0; i < outputHandlers.length; i++) 
				try {
					outputOperations[i].get();
				} catch (CancellationException | ExecutionException e) {
					continue;
				} catch (InterruptedException e) {
					break;
				}
		}
	}
	
}


package com.mjsd.simpleneuralnetwork;

import com.google.gson.Gson;
import com.google.gson.annotations.JsonAdapter;

import com.mjsd.simpleneuralnetwork.NetworkLayout.NetworkLayer;
import com.mjsd.simpleneuralnetwork.Serialization.NetworkSerializer;
import com.mjsd.simpleneuralnetwork.exceptions.DimensionsMismatchException;
import com.mjsd.simpleneuralnetwork.gson.*;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;

import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

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

	protected OutputHandler[] outputHandlers;
	protected InputProvider[] inputProviders;

	final private Object SYNCH_OBJECT = new Object();
	protected Runnable inputProtocol = this::getInputsLinear,
					   outputProtocol = this::handleOutputsLinear,
					   forwardPassProtocol = this::forwardPassLinear;

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
	 * @implNote This constructor does not check the given values and should only be used if the inputs are guaranteed to be valid.
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
			forwardPassProtocol.run();
			outputProtocol.run();
		}
	}

	protected void forwardPassLinear(){
		applyLayerModifiers(inputs, inputNormalizer, inputActivation);

		double[] previousLayer = inputs;

		for(int layer = 0; layer < hiddenLayers.length; layer++){
			NeuralNetworkTools.dotSequence(previousLayer, weights[layer], hiddenLayers[layer]);
			NeuralNetworkTools.vectorSum(hiddenLayers[layer], biases[layer], hiddenLayers[layer]);
			applyLayerModifiers(hiddenLayers[layer], hiddenNormalizers[layer], hiddenActivations[layer]);
			previousLayer = hiddenLayers[layer];
		}

		NeuralNetworkTools.dotSequence(previousLayer, weights[OUTPUT_LAYER], outputs);
		NeuralNetworkTools.vectorSum(outputs, biases[OUTPUT_LAYER], outputs);
		applyLayerModifiers(outputs, outputNormalizer, outputActivation);
	}

	protected void getInputsLinear(){
		for (int i = 0; i < inputs.length; i++)
			inputs[i] = inputProviders[i].orElse(inputs[i]);
	}

	protected void handleOutputsLinear(){
		for (int i = 0; i < outputHandlers.length; i++)
			outputHandlers[i].handle(outputs[i]);
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

		return (NeuralNetworkTools.haveSameLayout(this, other) &&
		   		NeuralNetworkTools.haveSameWeights(this, other) &&
		   		NeuralNetworkTools.haveSameBiases(this, other));
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

	public void saveBinary(String filePath, boolean overwrite) throws FileAlreadyExistsException, IOException, SecurityException, NullPointerException{
		NetworkSerializer serializer = new NetworkSerializer();
		serializer.save(this, filePath, overwrite);
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
		if (Objects.requireNonNull(values, "Value array is null.").length != this.inputs.length)
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




	/*******************************************************************************************************************
	************************************************* Static Methods ***************************************************
	*******************************************************************************************************************/

	protected static void applyLayerModifiers(double[] layer, InputNormalizer normalizer, ActivationFunction activationFunction){
		normalizer.normalize(layer);
		activationFunction.applyAll(layer, layer);
	}

	@JsonAdapter(ActivationFunctionAdapter.class)
	@FunctionalInterface
	public interface ActivationFunction{
		public double apply(double[] layer, int index);

		public default double[] applyAll(double[] layer){
			double[] applied = new double[layer.length];
			
			applyAll(layer, applied);

			return applied;
		}

		public default void applyAll(double[] layer, double[] destination){
			for(int i = 0; i < layer.length; i++)
				destination[i] = apply(layer, i);
		}

        public static ActivationFunction ensureFunction(ActivationFunction function){
            return (function == null) ? ActivationFunctions.LINEAR : function;
        }
	}

	@JsonAdapter(InputNormalizerAdapter.class)
	@FunctionalInterface
	public interface InputNormalizer {
		public void normalize(double[] values);

        public static InputNormalizer ensureNormalizer(InputNormalizer normalizer){
            return (normalizer == null) ? InputNormalizers.NONE : normalizer;
        }
	}

	@FunctionalInterface
	public interface OutputHandler {
		final static OutputHandler NO_HANDLER = x -> {};

		public void handle(double output);

		public static OutputHandler ensureHandler(OutputHandler handler){
			return (handler == null) ? NO_HANDLER : handler;
		}

		public static OutputHandler none(){
			return NO_HANDLER;
		}
	}

	@FunctionalInterface
	public interface InputProvider {
		final static InputProvider NO_PROVIDER = x -> x;

		public double orElse(double value);

		private static InputProvider ensureProvider(InputProvider provider){
			return (provider == null) ? NO_PROVIDER : provider;
		}

		public static InputProvider none(){
			return NO_PROVIDER;
		}
	}
	
}

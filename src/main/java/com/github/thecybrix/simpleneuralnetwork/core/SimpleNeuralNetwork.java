
package com.github.thecybrix.simpleneuralnetwork.core;

import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayout.NetworkLayer;
import com.github.thecybrix.simpleneuralnetwork.exceptions.DimensionsMismatchException;
import com.github.thecybrix.simpleneuralnetwork.serialization.binary.NetworkSerializer;
import com.github.thecybrix.simpleneuralnetwork.serialization.json.*;
import com.google.gson.Gson;
import com.google.gson.annotations.JsonAdapter;

import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

@JsonAdapter(SimpleNeuralNetworkAdapter.class)
public class SimpleNeuralNetwork {

	final protected int OUTPUT_LAYER;

	private HashMap<String, String> metadata = null;

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

	final private Object SYNCH_OBJECT = new Object();
	protected Runnable forwardPassProtocol = this::forwardPassLinear;

	protected SimpleNeuralNetwork(SimpleNeuralNetwork template) throws NullPointerException{
		this.OUTPUT_LAYER = template.OUTPUT_LAYER;

		this.inputs = new double[template.inputs.length];
		this.inputNormalizer = template.inputNormalizer;
		this.inputActivation = template.inputActivation;

		this.outputs = new double[template.outputs.length];
		this.outputNormalizer = template.outputNormalizer;
		this.outputActivation = template.outputActivation;
		
		this.hiddenLayers = new double[template.hiddenLayers.length][];
		this.hiddenActivations = new ActivationFunction[template.hiddenActivations.length];
		this.hiddenNormalizers = new InputNormalizer[template.hiddenNormalizers.length];

		for (int layer = 0; layer < template.hiddenLayers.length; layer++){
			this.hiddenLayers[layer] = new double[template.hiddenLayers[layer].length];
			this.hiddenActivations[layer] = template.hiddenActivations[layer].copyOrReuse();
			this.hiddenNormalizers[layer] = template.hiddenNormalizers[layer].copyOrReuse();
		}


		this.weights = NeuralNetworkTools.deepCopy(template.weights);
		this.biases = NeuralNetworkTools.deepCopy(template.biases);
	}

	
	protected SimpleNeuralNetwork(NetworkLayout layout) throws NullPointerException {
		Objects.requireNonNull(layout);

		NetworkLayer inputLayerLayout = layout.getInputLayer(),
					 outputLayerLayout = layout.getOutputLayer();
		List<NetworkLayer> hiddenLayerLayouts = layout.getHiddenLayers();

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
	protected SimpleNeuralNetwork(NetworkLayout layout, double[][][] weights, double[][] biases) throws NullPointerException {
		this.weights = weights;
		this.biases = biases;

		NetworkLayer inputLayerLayout = layout.getInputLayer(),
					 outputLayerLayout = layout.getOutputLayer();
		List<NetworkLayer> hiddenLayerLayouts = layout.getHiddenLayers();

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

	public double[] predict(double[] inputs) throws DimensionsMismatchException, NullPointerException{
		setInputs(inputs);
		forwardPass();
		return outputs;
	}

	public void forwardPass(){
		synchronized(SYNCH_OBJECT){
			forwardPassProtocol.run();
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

	public Map<String, String> getMetadata(){
		if (metadata == null)
			return Collections.emptyMap();
		else
			return Collections.unmodifiableMap(metadata);
	}

	public Optional<String> getMetadata(String key){
		if (metadata == null || !metadata.containsKey(key))
			return Optional.empty();
		else
			return Optional.of(metadata.get(key));
	}

	public double getInput(int index) throws IndexOutOfBoundsException {
		return inputs[index];
	}

	public double[] getInputLayer() {
		return inputs;
	}

	public double getOutput(int index) throws IndexOutOfBoundsException {
		return outputs[index];
	}

	/**
	 * @return a copy of the network's output array.
	 */
	public double[] getOutputs(){
		return Arrays.copyOf(outputs, outputs.length);
	}

	/**
	 * @return the network's output array in it's current state. This array will be modified by any calls to {@link #forwardPass()}, and any modification to it will be reflected in {@link #getOutput(int)} and {@link #getOutputs()}.
	 */
	public double[] getOutputLayer() {
		return outputs;
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

	public void setMetadata(Map<String, String> data){
		if(data == null || data.isEmpty())
			metadata = null;
		else
			metadata = new HashMap<>(data);
	}

	public void clearMetadata(){
		metadata = null;
	}

	public String putMetadata(String key, String value){
		if (metadata == null)
			metadata = new HashMap<>();
		
		return metadata.put(key, value);
	}

	public String removeMetadata(String key){
		if(metadata == null) return null;
		String value = metadata.remove(key);
		if(metadata.size() == 0) metadata = null;
		return value;
	}

	public void setInputs(double[] values) throws NullPointerException, DimensionsMismatchException {
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
	
}

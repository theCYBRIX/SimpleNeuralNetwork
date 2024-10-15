package com.github.thecybrix.simpleneuralnetwork.core;

import java.util.Objects;

import com.github.thecybrix.simpleneuralnetwork.exceptions.DimensionsMismatchException;
import com.github.thecybrix.simpleneuralnetwork.serialization.json.MutableNeuralNetworkAdapter;
import com.google.gson.Gson;
import com.google.gson.annotations.JsonAdapter;

@JsonAdapter(MutableNeuralNetworkAdapter.class)
public class MutableNeuralNetwork extends SimpleNeuralNetwork{

	public MutableNeuralNetwork(NetworkLayout layout) throws NullPointerException {
		super(layout);
	}

	public MutableNeuralNetwork(SimpleNeuralNetwork template) throws NullPointerException {
		super(template);
	}

	protected MutableNeuralNetwork(NetworkLayout layout, double[][][] weights, double[][] biases, OutputHandler[] outputHandlers, InputProvider[] inputProviders) throws DimensionsMismatchException, NullPointerException {
		super(layout, weights, biases, outputHandlers, inputProviders);
	}

	@Override
	public MutableNeuralNetwork copy() {
		return new MutableNeuralNetwork(this);
	}

	@Override
	public String toJson(Gson gson) {
        return gson.toJson(this, MutableNeuralNetwork.class);
	}

	
	/*******************************************************************************************************************
	***************************************************** Getters ******************************************************
	*******************************************************************************************************************/

	@Override
	public double[][] getHiddenLayers() {
		return super.getHiddenLayers();
	}

	/*******************************************************************************************************************
	***************************************************** Setters ******************************************************
	*******************************************************************************************************************/

	public void setActivationFunction(int layerIndex, ActivationFunction activationFunction) {
		activationFunction = ActivationFunction.ensureFunction(activationFunction);
		synchronized (hiddenActivations) {
			hiddenActivations[layerIndex] = activationFunction;
		}
	}




	/*******************************************************************************************************************
	************************************************ Weight Management *************************************************
	*******************************************************************************************************************/

	/**
	 * @return The 3-dimensional array containing the weights of this network; consisting of {@code weights[layer][node][weight]}.
	 * @implNote This method is intended for quick read access to the weights of this network. Any modifications to the array will be reflected in the original network.
	 * @see #getWeights()
	 */
	public double[][][] retrieveWeightsArray()  {
		return weights;
	}

	/**
	 * @param layerIndex The index of the layer from which to select the indexed node. {@code layerIndex = 0} refers to the first hidden layer, and {@code layerIndex = numHiddenLayers} refers to the output layer.
	 * @return a 2-dimensional array containing the weights of the nodes in the selected layer, consisting of {@code weights[node][weight]}. 
	 * @throws ArrayIndexOutOfBoundsException
	 */
	public double[][] getWeights(int layerIndex) throws ArrayIndexOutOfBoundsException {
		return weights[layerIndex];
	}


	/**
	 * @param layerIndex The index of the layer from which to select the indexed node. {@code layerIndex = 0} refers to the first hidden layer, and {@code layerIndex = numHiddenLayers} refers to the output layer.
	 * @param nodeIndex The index of the node from which to retrieve the weights.
	 * @return an array containing the weights of the selected node.
	 * @throws ArrayIndexOutOfBoundsException
	 */
	public double[] getWeights(int layerIndex, int nodeIndex) throws ArrayIndexOutOfBoundsException {
		return weights[layerIndex][nodeIndex];
	}


	/**
	 * @param layerIndex The index of the layer from which to select the indexed node. {@code layerIndex = 0} refers to the first hidden layer, and {@code layerIndex = numHiddenLayers} refers to the output layer.
	 * @param nodeIndex The index of the node from which to retrieve the weight.
	 * @param weightIndex The index of the weight to return.
	 * @return a double representing the value of the selected weight. 
	 * @throws ArrayIndexOutOfBoundsException
	 */
	public double getWeight(int layerIndex, int nodeIndex, int weightIndex) throws ArrayIndexOutOfBoundsException {
		return this.weights[layerIndex][nodeIndex][weightIndex];
	}

	public void setWeights(double[][][] values) throws DimensionsMismatchException, NullPointerException {
		setWeightsUnchecked(NeuralNetworkTools.ensureValidWeightArray(LAYOUT, values));
	}

	protected void setWeightsUnchecked(double[][][] values) throws ArrayIndexOutOfBoundsException, NullPointerException {
		for (int layer = 0; layer < values.length; layer++)
			for (int node = 0; node < values[layer].length; node++)
				for (int weight = 0; weight < values[layer][node].length; weight++)
					this.weights[layer][node][weight] = values[layer][node][weight];
	}

	public void setWeights(int layerIndex, double[][] values) throws DimensionsMismatchException, NullPointerException {
		Objects.requireNonNull(values);

		if (values.length != this.weights[layerIndex].length)
			throw new DimensionsMismatchException(
					"Number of nodes in Layer " + layerIndex + " do not match. Weight array contains " + values.length
							+ " node/s, whereas the network has " + this.weights[layerIndex].length + " node/s.");

		for (int node = 0; node < this.weights[layerIndex].length; node++)
			if (values[node].length != this.weights[layerIndex][node].length)
				throw new DimensionsMismatchException("Number of weights in node " + node + " of Layer " + layerIndex
						+ " do not match. Weight array contains " + values[node].length
						+ " node/s, whereas the network has " + this.weights[layerIndex][node].length + " wight/s.");

		setWeightsUnchecked(layerIndex, values);
	}

	protected void setWeightsUnchecked(int layerIndex, double[][] values) throws ArrayIndexOutOfBoundsException, NullPointerException {
		for (int node = 0; node < values.length; node++)
			for (int weight = 0; weight < values[node].length; weight++)
				this.weights[layerIndex][node][weight] = values[node][weight];
	}

	public void setWeights(int layerIndex, int nodeIndex, double[] values) throws DimensionsMismatchException, NullPointerException {
		Objects.requireNonNull(values);

		if (values.length != this.weights[layerIndex][nodeIndex].length)
			throw new DimensionsMismatchException("Number of weights in node " + nodeIndex + " of Layer " + layerIndex
					+ " do not match. Weight array contains " + values.length + " nodes, whereas the network has "
					+ this.weights[layerIndex][nodeIndex].length + " wight/s.");

		setWeightsUnchecked(layerIndex, nodeIndex, values);
	}

	protected void setWeightsUnchecked(int layerIndex, int nodeIndex, double[] values) throws ArrayIndexOutOfBoundsException, NullPointerException {
		for (int weight = 0; weight < values.length; weight++)
			this.weights[layerIndex][nodeIndex][weight] = values[weight];
	}

	public void setWeight(int layerIndex, int nodeIndex, int weightIndex, double value) throws IndexOutOfBoundsException {
		this.weights[layerIndex][nodeIndex][weightIndex] = value;
	}



	/*******************************************************************************************************************
	************************************************* Bias Management **************************************************
	*******************************************************************************************************************/
	

	/**
	 * @return The 2-dimensional array containing the biases of this network; consisting of {@code biases[layer][node]}.
	 * @implNote This method is intended for quick read access to the biases of this network. Any modifications to the array will be reflected in the original network.
	 * @see #getBiases()
	 */
	public double[][] retrieveBiasesArray(){
		return biases;
	}

	public double[] getBiases(int layerIndex) throws IndexOutOfBoundsException {
		return this.biases[layerIndex];
	}

	public double getBias(int layerIndex, int nodeIndex) throws IndexOutOfBoundsException {
		return this.biases[layerIndex][nodeIndex];
	}
	
	public void setBiases(double[][] values) throws DimensionsMismatchException, NullPointerException {
		setBiasesUnchecked(NeuralNetworkTools.ensureValidBiasArray(LAYOUT, values));
	}
	
	protected void setBiasesUnchecked(double[][] values) throws ArrayIndexOutOfBoundsException, NullPointerException {
		for (int layer = 0; layer < values.length; layer++)
			for (int node = 0; node < values[layer].length; node++)
				this.biases[layer][node] = values[layer][node];
	}

	public void setBiases(int layerIndex, double[] values) throws DimensionsMismatchException, NullPointerException {
		Objects.requireNonNull(values);

		if (values.length != this.biases[layerIndex].length)
			throw new DimensionsMismatchException(
					"Number of nodes in Layer " + layerIndex + " do not match. Bias array contains " + values.length
							+ " node/s, whereas the network has " + this.biases[layerIndex].length + " node/s.");

		for (int node = 0; node < this.biases[layerIndex].length; node++)
			this.biases[layerIndex][node] = values[node];
	}

	protected void setBiasesUnchecked(int layerIndex, double[] values) throws ArrayIndexOutOfBoundsException, NullPointerException {
		for (int node = 0; node < values.length; node++)
			this.biases[layerIndex][node] = values[node];
	}

	public void setBias(int layerIndex, int nodeIndex, double value) throws IndexOutOfBoundsException {
		this.biases[layerIndex][nodeIndex] = value;
	}

	@Override
	public NeuralNetworkBuilder<? extends MutableNeuralNetwork> newBuilder(){
		return new MutableNeuralNetworkBuilder(this);
	}

}
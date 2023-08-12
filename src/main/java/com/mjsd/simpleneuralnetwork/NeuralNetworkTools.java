package com.mjsd.simpleneuralnetwork;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.function.Supplier;

import com.mjsd.simpleneuralnetwork.exceptions.ArraySizeMismatchException;
import com.mjsd.simpleneuralnetwork.exceptions.DimensionsMismatchException;
import com.mjsd.simpleneuralnetwork.exceptions.LayoutMismatchException;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.NetworkLayout.NetworkLayer;

public abstract class NeuralNetworkTools {

	final private static Random RANDOM = new Random();

	/**
	 * @return A number "N" such that {@code (origin - maxDeviation) < N < (origin + maxDeviation) }.
	 */
	private static double randomOffset(double origin, double maxDeviation, Random random) {
		return origin + random.nextDouble() * (random.nextBoolean() ? maxDeviation : -maxDeviation);
	}

	/**
	 * @return A number "N" such that {@code (origin - maxDeviation) < N < (origin + maxDeviation) }.
	 */
	private static double randomOffset(double origin, double maxDeviation) {
		return randomOffset(origin, maxDeviation, RANDOM);
	}


	/*******************************************************************************************************************
	******************************************* Weight and Bias Management *********************************************
	*******************************************************************************************************************/

	public static <T extends MutableNeuralNetwork> T copyWeightsAndBiases(T from, T to) throws DimensionsMismatchException, NullPointerException{
		requireSameDimensions(from, to);
		copyWeightsAndBiasesUnchecked(from, to);
		return to;
	}

	public static <T extends MutableNeuralNetwork> T copy(T from, Supplier<T> to) throws DimensionsMismatchException, NullPointerException{
		return copyWeightsAndBiases(from, to.get());
	}

	protected static void copyWeightsAndBiasesUnchecked(MutableNeuralNetwork from, MutableNeuralNetwork to) throws NullPointerException{
		to.weights = NeuralNetworkTools.deepCopy(from.weights);
		to.biases = NeuralNetworkTools.deepCopy(from.biases);
	}


	public static <E extends MutableNeuralNetwork> E shiftWeightsAndBiases(E network, double maxWeightOffset, double maxBiasOffset) throws NullPointerException {
		double[][][] weights = network.weights;
		double[][] biases = network.biases;
		for (int layer = 0; layer < weights.length; layer++)
			for (int n = 0; n < weights[layer].length; n++) {
				biases[layer][n] += randomOffset(biases[layer][n], maxBiasOffset);
				for (int i = 0; i < weights[layer][n].length; i++)
					weights[layer][n][i] += randomOffset(weights[layer][n][i], maxWeightOffset);
			}
		return network;
	}

	public static <E extends MutableNeuralNetwork> E randomizeWeightsAndBiases(E network, double weightOffset, double biasOffset) throws NullPointerException {
		double[][][] weights = network.weights;
		double[][] biases = network.biases;
		for (int layer = 0; layer < weights.length; layer++)
			for (int n = 0; n < weights[layer].length; n++) {
				biases[layer][n] = randomOffset(biases[layer][n], biasOffset);
				for (int i = 0; i < weights[layer][n].length; i++)
					weights[layer][n][i] = randomOffset(weights[layer][n][i], weightOffset);
			}
		return network;
	}

	public static <E extends MutableNeuralNetwork> E adjustWeight(E network, int layer, int node, int weight, double adjustAmount) throws NullPointerException {
		double[][][] weights = network.weights;
		weights[layer][node][weight] += adjustAmount;
		return network;
	}

	public static <E extends MutableNeuralNetwork> E adjustWeights(E network, double maxOffset) throws NullPointerException {
		double[][][] weights = network.weights;
		for (int layer = 0; layer < weights.length; layer++)
			for (int w = 0; w < weights[layer].length; w++)
				for (int i = 0; i < weights[layer][w].length; i++)
					weights[layer][w][i] = randomOffset(weights[layer][w][i], maxOffset);

		return network;
	}

	public static <E extends MutableNeuralNetwork> E randomizeWeights(E network, double maxOffset) throws NullPointerException {
		double[][][] weights = network.weights;
		for (int layer = 0; layer < weights.length; layer++)
			for (int w = 0; w < weights[layer].length; w++)
				for (int i = 0; i < weights[layer][w].length; i++)
					weights[layer][w][i] = randomOffset(weights[layer][w][i], maxOffset);

		return network;
	}

	public static <E extends MutableNeuralNetwork> E adjustBias(E network, int layer, int node, double adjustAmount) throws NullPointerException {
		network.biases[layer][node] += adjustAmount;
		return network;
	}

	public <E extends MutableNeuralNetwork> E randomizeBiases(E network, double maxOffset) throws NullPointerException {
		double[][] biases = network.biases;
		for (int layer = 0; layer < biases.length; layer++)
			for (int b = 0; b < biases[layer].length; b++)
				biases[layer][b] = randomOffset(biases[layer][b], maxOffset);
		return network;
	}
    
	public static double categoricalCrossEntropy(double[] targetOutput, double[] actualOutput) throws ArraySizeMismatchException, NullPointerException {
        if (Objects.requireNonNull(targetOutput).length != Objects.requireNonNull(actualOutput).length)
            throw new ArraySizeMismatchException("Arrays must be of equal size. (" + targetOutput.length + " != " + actualOutput.length + ")");

        double loss = 0;

        for (int i = 0; i < targetOutput.length; i++)
            loss += Math.log(actualOutput[i] * targetOutput[i]);

        loss = (-loss);

        return loss;
    }

	/**
	 * Crosses two networks weights at a random point within each layer.
	 * 
	 * @param parent1
	 * @param parent2
	 * @return An array of length 2 containing two children, each with a mixture of
	 *         features from either parent.
	 */
	public static <E extends MutableNeuralNetwork> E layerwiseCrossover(E parent1, E parent2, Supplier<E> child) throws LayoutMismatchException, NullPointerException {
		E childNetwork = child.get();

		requireSameLayout(parent1, parent2, childNetwork);

		List<NetworkLayer> hiddenLayers = parent1.getLayout().getHiddenLayers();

		int[] splitPoints = new int[hiddenLayers.size()];

		/**
		* A random split-point in range [0.25, 0.75) or, [25%, 75%) of each layer.
		* If the value is 0.25, child1 receives 25% of parent1, and 75% of parent two; child2 receives the opposite.
		*/
		double splitFraction;

		for(int layer = 0; layer < splitPoints.length; layer++){
			splitFraction = 0.25 + 0.5 * Math.random();
			splitPoints[layer] = (int)Math.round(hiddenLayers.get(layer).getNodeCount() * splitFraction);
		}
		
		return crossover(parent1, parent2, childNetwork, splitPoints);
	}

	/**
	 * Crosses two networks weights at a random point within each layer.
	 * 
	 * @param parent1
	 * @param parent2
	 * @return An array of length 2 containing two children, each with a mixture of
	 *         features from either parent.
	 */
	public static <E extends MutableNeuralNetwork> E singlePointCrossover(E parent1, E parent2, Supplier<E> child) throws LayoutMismatchException, NullPointerException {
		E childNetwork = child.get();

		requireSameLayout(parent1, parent2, childNetwork);

		List<NetworkLayer> hiddenLayers = parent1.getLayout().getHiddenLayers();

		int[] splitPoints = new int[hiddenLayers.size()];

		/**
		* A random split-point in range [0.25, 0.75) or, [25%, 75%) of each layer.
		* If the value is 0.25, child1 receives 25% of parent1, and 75% of parent two; child2 receives the opposite.
		*/
		double splitFraction = 0.25 + 0.5 * Math.random();

		for(int layer = 0; layer < splitPoints.length; layer++)
			splitPoints[layer] = (int)Math.round(hiddenLayers.get(layer).getNodeCount() * splitFraction);
		
		return crossover(parent1, parent2, childNetwork, splitPoints);
	}

	private static <E extends MutableNeuralNetwork> E crossover(E parent1, E parent2, E child, int[] splitPoints) throws NullPointerException{

		try {
			int node;
			List<NetworkLayer> hiddenLayers = child.getLayout().getHiddenLayers();
			for (int layer = 0; layer < hiddenLayers.size(); layer++) {
				for (node = 0; node < splitPoints[layer]; node++) {
					child.setWeights(layer, node, parent1.getWeights(layer, node));
					child.setBias(layer, node, parent1.getBias(layer, node));
				}
				int nodesInLayer  = hiddenLayers.get(layer).getNodeCount();
				for (; node < nodesInLayer; node++) {
					child.setWeights(layer, node, parent2.getWeights(layer, node));
					child.setBias(layer, node, parent2.getBias(layer, node));
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		return child;
	}

	public static <E extends MutableNeuralNetwork> E getRandomizedNetwork(Supplier<E> networkSource) throws NullPointerException{
		Objects.requireNonNull(networkSource, "Network Supplier is null.");
		return randomizeWeightsAndBiases(networkSource.get(), 1, 1);
	}

	public static <E extends MutableNeuralNetwork> ArrayList<E> getRandomizedNetworks(int numNetworks, Supplier<E> networkSource) throws IllegalArgumentException, NullPointerException{
		Objects.requireNonNull(networkSource, "Network Supplier is null.");

		ArrayList<E> networks = new ArrayList<>(numNetworks);
		for(int i = 0; i < numNetworks; i++)
			networks.add(randomizeWeightsAndBiases(networkSource.get(), 1, 1));

		return networks;
	}

	public static <E extends MutableNeuralNetwork> E getMutation(E baseNetwork, Supplier<E> networkSource, double maxWeightDeviation, double maxBiasDeviation) throws IllegalArgumentException, NullPointerException{
		Objects.requireNonNull(networkSource, "Network Supplier is null.");
		return getMutation(networkSource, baseNetwork.getWeights(), baseNetwork.getBiases(), maxWeightDeviation, maxBiasDeviation);
	}

	public static <E extends MutableNeuralNetwork> List<E> getMutations(E baseNetwork, int numMutations, Supplier<E> networkSource, double maxWeightDeviation, double maxBiasDeviation) throws IllegalArgumentException, NullPointerException{
		Objects.requireNonNull(networkSource, "Network Supplier is null.");
		ArrayList<E> mutations = new ArrayList<>(numMutations);
		double[][][] weights = baseNetwork.getWeights();
		double[][] biases = baseNetwork.getBiases();
		
		E mutation;
		for(int i = 0; i < numMutations; i++){
			mutation = getMutation(networkSource, weights, biases, maxWeightDeviation, maxBiasDeviation);
			mutations.add(mutation);
		}

		return mutations;
	}

	private static <E extends MutableNeuralNetwork> E getMutation(Supplier<E> networkSource, double[][][] weights, double[][] biases, double maxWeightDeviation, double maxBiasDeviation) throws IllegalArgumentException, NullPointerException{
		E mutation = networkSource.get();
		mutation.setWeights(weights);
		mutation.setBiases(biases);
		shiftWeightsAndBiases(mutation, maxWeightDeviation, maxBiasDeviation);

		return mutation;
	}

	public static <E extends MutableNeuralNetwork> E randomMutation(E network, double maxWeightDeviation, double maxBiasDeviation, Random random){
		double[][][] weights = network.weights;
		double[][] biases = network.biases;

		for(int layer = 0; layer < weights.length; layer++)
			for(int node = 0; node < weights[layer].length; node++){
				if(random.nextBoolean()) biases[layer][node] = randomOffset(biases[layer][node], maxBiasDeviation, random);
				for(int weight = 0; weight < weights[layer][node].length; weight++)
					if(random.nextBoolean()) weights[layer][node][weight] = randomOffset(weights[layer][node][weight], maxWeightDeviation, random);
			}


		return network;
	}

	public static <E extends MutableNeuralNetwork> E randomMutation(E network, double maxWeightDeviation, double maxBiasDeviation){
		return randomMutation(network, maxWeightDeviation, maxBiasDeviation, RANDOM);
	}

	public static <E extends MutableNeuralNetwork> Set<E> createUniqueMutations(E baseNetwork, double maxWeightDeviation, double maxBiasDeviation, Supplier<E> networkSource, int numMutations) throws IllegalArgumentException, NullPointerException{
		HashSet<E> mutations = new HashSet<>(numMutations);
		
		double[][][] weights = baseNetwork.getWeights();
		double[][] biases = baseNetwork.getBiases();
		E mutation;

		while(mutations.size() < numMutations){
			mutation = networkSource.get();
			mutation.setWeights(weights);
			mutation.setBiases(biases);
			shiftWeightsAndBiases(mutation, maxWeightDeviation, maxBiasDeviation);
			mutations.add(mutation);
		}

		return mutations;
	}

	public static <E extends MutableNeuralNetwork> Set<E> createUniqueChildren(E parent1, E parent2, Supplier<E> childSource, int numChildren) throws IllegalArgumentException, NullPointerException{
		HashSet<E> children = new HashSet<>(numChildren);

		while(children.size() < numChildren){ 
			children.add(NeuralNetworkTools.singlePointCrossover(parent1, parent2, childSource));
		}

		return children;
	}

	public static double[][] ensureValidBiasArray(NetworkLayout layout, double[][] biases) throws DimensionsMismatchException, NullPointerException {
		Objects.requireNonNull(biases);
		int numBiasLayers = layout.getHiddenLayers().size() + 1;
		ArrayList<NetworkLayer> hiddenLayers = new ArrayList<>(numBiasLayers);
		hiddenLayers.addAll(layout.getHiddenLayers());
		hiddenLayers.add(layout.getOutputLayer());

		if (biases.length != numBiasLayers)
			throw new DimensionsMismatchException("Number of layers does not match. Bias array contains " + biases.length
					+ " layer/s, whereas the network has " + numBiasLayers + " layer/s.");

		for (int layer = 0; layer < biases.length; layer++)
			if (biases[layer].length != hiddenLayers.get(layer).getNodeCount())
				throw new DimensionsMismatchException("Number of nodes in Layer " + layer + " do not match. Bias array contains "
						+ biases[layer].length + " node/s, whereas the network has " + hiddenLayers.get(layer).getNodeCount()
						+ " node/s.");

		return biases;
	}

	public static double[][][] ensureValidWeightArray(NetworkLayout layout, double[][][] weights) throws DimensionsMismatchException, NullPointerException {
		Objects.requireNonNull(weights);
		int numWeightLayers = layout.getHiddenLayers().size() + 1;
		ArrayList<NetworkLayer> hiddenLayers = new ArrayList<>(numWeightLayers);
		hiddenLayers.addAll(layout.getHiddenLayers());
		hiddenLayers.add(layout.getOutputLayer());

		if (weights.length != numWeightLayers)
			throw new DimensionsMismatchException("Number of weight layers does not match. Weight array contains " + weights.length
					+ " layers, whereas the network layout has " + numWeightLayers + " layers.");

		int numCurrentLayerNodes;
		for (int layer = 0; layer < numWeightLayers; layer++){
			numCurrentLayerNodes = hiddenLayers.get(layer).getNodeCount();
			if (weights[layer].length != numCurrentLayerNodes)
				throw new DimensionsMismatchException("Number of nodes in Layer " + layer
						+ " do not match. Weight array contains " + weights[layer].length
						+ " node/s, whereas the network layout has " + numCurrentLayerNodes + " node/s.");
		}

		int numPrevLayerNodes = layout.getInputLayer().getNodeCount();

		for (int layer = 0; layer < numWeightLayers; layer++){
			numCurrentLayerNodes = hiddenLayers.get(layer).getNodeCount();
			for (int node = 0; node < numCurrentLayerNodes; node++){
				if (weights[layer][node].length != numPrevLayerNodes)
					throw new DimensionsMismatchException("Number of weights in node " + node + " of Layer " + layer
							+ " do not match. Weight array contains " + weights[layer][node].length
							+ " weight/s, whereas the network layout has " + numPrevLayerNodes + " wight/s.");
			}
			numPrevLayerNodes = numCurrentLayerNodes;
		}
		
		return weights;
	}

	public static boolean haveSameActivationFunction(SimpleNeuralNetwork network1, SimpleNeuralNetwork network2, int layer) throws NullPointerException {
		NetworkLayer network1Layer = network1.getHiddenLayerLayout(layer);
		NetworkLayer network2Layer = network2.getHiddenLayerLayout(layer);

		return network1Layer.getActivationFunction().equals(network2Layer.getActivationFunction());
	}


	public static void requireSameLayout(SimpleNeuralNetwork network1, SimpleNeuralNetwork network2, SimpleNeuralNetwork... networks) throws LayoutMismatchException, NullPointerException{
		if ( !haveSameLayout(network1, network2, networks) )
			throw new LayoutMismatchException("Networks have different layouts.");
	}


	public static void requireSameLayout(SimpleNeuralNetwork network1, SimpleNeuralNetwork network2) throws LayoutMismatchException, NullPointerException{
		if ( !haveSameLayout(network1, network2) )
			throw new LayoutMismatchException("Networks have different layouts.");
	}

	public static boolean haveSameLayout(SimpleNeuralNetwork network1, SimpleNeuralNetwork network2, SimpleNeuralNetwork... networks) throws NullPointerException {
		
		if(!haveSameLayout(network1, network2)) return false;

		SimpleNeuralNetwork previousNetwork = network2;
		for (SimpleNeuralNetwork network : networks) {
			if(!haveSameLayout(previousNetwork, network)) return false;
			previousNetwork = network;
		}

		return true;
	}

	private static boolean haveSameLayout(SimpleNeuralNetwork network1, SimpleNeuralNetwork network2) {		
		return network1.getLayout().equals(network2.getLayout());
	}

	public static void requireSameDimensions(SimpleNeuralNetwork network1, SimpleNeuralNetwork network2) throws DimensionsMismatchException, NullPointerException {
		if(!haveSameDimensions(network1, network2)) throw new DimensionsMismatchException("Network dimensions are not equal.");
	}

	public static void requireSameDimensions(SimpleNeuralNetwork network1, SimpleNeuralNetwork network2, SimpleNeuralNetwork... networks) throws DimensionsMismatchException, NullPointerException {
		if(!haveSameDimensions(network1, network2, networks)) throw new DimensionsMismatchException("Network dimensions are not equal.");
	}

	public static boolean haveSameDimensions(SimpleNeuralNetwork network1, SimpleNeuralNetwork network2, SimpleNeuralNetwork... networks) throws NullPointerException {
		if(!haveSameDimensions(network1, network2)) return false;
		
		SimpleNeuralNetwork previousNetwork = network2;
		for (SimpleNeuralNetwork network : networks) {
			if(!haveSameDimensions(previousNetwork, network)) return false;
			previousNetwork = network;
		}

		return true;
	}

	private static boolean haveSameDimensions(SimpleNeuralNetwork network1, SimpleNeuralNetwork network2) throws NullPointerException {

		NetworkLayout layout1 = network1.getLayout(),
					  layout2 = network2.getLayout();

		if (layout1.getInputLayer().getNodeCount() != layout2.getInputLayer().getNodeCount())
			return false;

		if (layout1.getOutputLayer().getNodeCount() != layout2.getOutputLayer().getNodeCount())
			return false;

		List<NetworkLayer> hiddenLayers1 = layout1.getHiddenLayers(),
						   hiddenLayers2 = layout2.getHiddenLayers();
			
		if(hiddenLayers1.size() != hiddenLayers2.size())
			return false;
		
		Iterator<NetworkLayer> layerIterator1 = hiddenLayers1.iterator(),
							   layerIterator2 = hiddenLayers2.iterator();

		while(layerIterator1.hasNext() && layerIterator2.hasNext())
			if(layerIterator1.next().getNodeCount() != layerIterator2.next().getNodeCount())
				return false;

		return true;
	}

	/**
	 * Returns a List containing the specified number of randomly selected objects from the source list.this list between , and toIndex, exclusive.
	 * @param <T>
	 * @param source The source list from which to pick objects.
	 * @param outputLength The number of objects to pick from the source list.
	 * @param fromIndex The index (inclusive) after which to pick objects from the source list.
	 * @param toIndex The index (exclusive) before which to pick objects from the source list.
	 * @return A list containing randomly chosen objects from the source list.
	 * @throws IllegalArgumentException If {@code outputLength > (toIndex - fromIndex)}{@code outputLength > (toIndex - fromIndex)}.
	 */
	final protected static <T> List<T> pickRandom(List<T> source, int outputLength, int fromIndex, int toIndex) throws IllegalArgumentException{
		if(outputLength < 0) throw new IllegalArgumentException("Cannot return a list of length " + outputLength + ".");
		if(outputLength > (toIndex - fromIndex))
			throw new IllegalArgumentException("Output length (" + outputLength + ") greater than specified range [" + fromIndex + ", " + toIndex + ")");

		ArrayList<T> subList, randoms;

		subList = new ArrayList<>(source.subList(fromIndex, toIndex));
		randoms = new ArrayList<>(outputLength);

		Collections.shuffle(subList);

		for(int i = 0; i < outputLength; i++){
			randoms.add(subList.get(i));
		}

		return randoms;
	}

	final public static double[][] deepCopy(double[][] toCopy){
		double[][] copy = new double[toCopy.length][];
		for(int i = 0; i < toCopy.length; i++){
			copy[i] = new double[toCopy[i].length];
			for(int j = 0; j < toCopy[i].length; j++)
				copy[i][j] = toCopy[i][j];
		}
		return copy;
	}

	final public static double[][][] deepCopy(double[][][] toCopy){
		double[][][] copy = new double[toCopy.length][][];
		for(int i = 0; i < toCopy.length; i++){
			copy[i] = new double[toCopy[i].length][];
			for(int j = 0; j < toCopy[i].length; j++){
				copy[i][j] = new double[toCopy[i][j].length];
				for(int k = 0; k < toCopy[i][j].length; k++)
					copy[i][j][k] = toCopy[i][j][k];
			}
		}
		return copy;
	}
}

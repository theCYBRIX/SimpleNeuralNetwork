package com.github.thecybrix.simpleneuralnetwork.core;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.function.Supplier;

import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayout.NetworkLayer;
import com.github.thecybrix.simpleneuralnetwork.exceptions.DimensionsMismatchException;
import com.github.thecybrix.simpleneuralnetwork.exceptions.LayoutMismatchException;

final public class NeuralNetworkTools {
	final private static double DEFAULT_WEIGHT_ORIGIN = 0,
								DEFAULT_WEIGHT_DEVIATION = 0.5,

								DEFAULT_BIAS_ORIGIN = 0,
								DEFAULT_BIAS_DEVIATION = 1,
								
								DEFAULT_MUTATION_RATE = 0.5;

	private NeuralNetworkTools(){}

	/**
	 * @return A number "N" such that {@code (origin - maxDeviation) < N < (origin + maxDeviation) }.
	 */
	private static double randomOffset(double origin, double maxDeviation, Random random) throws NullPointerException {
		return origin + random.nextDouble() * (random.nextBoolean() ? maxDeviation : -maxDeviation);
	}

	/**
	 * @return A number "N" such that {@code (origin - maxDeviation) < N < (origin + maxDeviation) }.
	 */
	private static double randomOffset(double origin, double maxDeviation) {
		return randomOffset(origin, maxDeviation, RandomNumberGeneratorHolder.RANDOM);
	}

	protected static double dotProduct(double[] v1, double[] v2) {
		double product = 0;

		for (int i = 0; i < v1.length; i++)
			product += v1[i] * v2[i];

		return product;
	}

	protected static void dotSequence(double[] v1, double[][] crMatrix, double[] destination) {
		for (int column = 0; column < crMatrix.length; column++)
			destination[column] = dotProduct(v1, crMatrix[column]);
	}

	protected static void vectorSum(double[] v1, double[] v2, double[] destination) {
		for (int i = 0; i < v1.length; i++)
			destination[i] = v1[i] + v2[i];
    }


	/*******************************************************************************************************************
	******************************************* Weight and Bias Management *********************************************
	*******************************************************************************************************************/

	public static <T extends MutableNeuralNetwork> T copyWeightsAndBiases(Supplier<T> from, Supplier<T> to) throws DimensionsMismatchException, NullPointerException{
		return copyWeightsAndBiases(from.get(), to.get());
	}

	public static <T extends MutableNeuralNetwork> T copyWeightsAndBiases(T from, T to) throws DimensionsMismatchException, NullPointerException{
		requireSameDimensions(from, to);
		copyWeightsAndBiasesUnchecked(from, to);
		return to;
	}

	protected static <T extends MutableNeuralNetwork> void copyWeightsAndBiasesUnchecked(Supplier<T> from, Supplier<T> to) throws NullPointerException{
		copyWeightsAndBiasesUnchecked(from.get(), to.get());
	}

	protected static void copyWeightsAndBiasesUnchecked(MutableNeuralNetwork from, MutableNeuralNetwork to) throws NullPointerException{
		to.weights = NeuralNetworkTools.deepCopy(from.weights);
		to.biases = NeuralNetworkTools.deepCopy(from.biases);
	}


	public static <E extends MutableNeuralNetwork> E shiftWeightsAndBiases(E network, double maxWeightOffset, double maxBiasOffset) throws NullPointerException {
		return shiftWeightsAndBiases(network, maxWeightOffset, maxBiasOffset, RandomNumberGeneratorHolder.RANDOM);
	}

	public static <E extends MutableNeuralNetwork> E shiftWeightsAndBiases(E network, double maxWeightOffset, double maxBiasOffset, Random random) throws NullPointerException {
		double[][][] weights = network.weights;
		double[][] biases = network.biases;
		for (int layer = 0; layer < weights.length; layer++)
			for (int n = 0; n < weights[layer].length; n++) {
				biases[layer][n] += randomOffset(biases[layer][n], maxBiasOffset, random);
				for (int i = 0; i < weights[layer][n].length; i++)
					weights[layer][n][i] += randomOffset(weights[layer][n][i], maxWeightOffset, random);
			}
		return network;
	}

	public static <E extends MutableNeuralNetwork> E randomizeWeightsAndBiases(E network) throws NullPointerException {
		return randomizeWeightsAndBiases(network, RandomNumberGeneratorHolder.RANDOM);
	}

	public static <E extends MutableNeuralNetwork> E randomizeWeightsAndBiases(E network, Random random) throws NullPointerException {
		return randomizeWeightsAndBiases(network, DEFAULT_WEIGHT_ORIGIN, DEFAULT_BIAS_DEVIATION, DEFAULT_BIAS_ORIGIN, DEFAULT_BIAS_DEVIATION, random);
	}

	public static <E extends MutableNeuralNetwork> E randomizeWeightsAndBiases(E network, double weightOrigin, double wOffsetMagnitude, double biasOrigin, double bOffsetMagnitude) throws NullPointerException {
		return randomizeWeightsAndBiases(network, weightOrigin, wOffsetMagnitude, biasOrigin, bOffsetMagnitude, RandomNumberGeneratorHolder.RANDOM);
	}

	public static <E extends MutableNeuralNetwork> E randomizeWeightsAndBiases(E network, double weightOrigin, double wOffsetMagnitude, double biasOrigin, double bOffsetMagnitude, Random random) throws NullPointerException {
		double[][][] weights = network.weights;
		double[][] biases = network.biases;
		for (int layer = 0; layer < weights.length; layer++)
			for (int n = 0; n < weights[layer].length; n++) {
				biases[layer][n] = randomOffset(biasOrigin, bOffsetMagnitude, random);
				for (int i = 0; i < weights[layer][n].length; i++)
					weights[layer][n][i] = randomOffset(weightOrigin, wOffsetMagnitude, random);
			}
		return network;
	}

	public static <E extends MutableNeuralNetwork> E adjustWeight(E network, int layer, int node, int weight, double adjustAmount) throws NullPointerException, ArrayIndexOutOfBoundsException {
		network.weights[layer][node][weight] += adjustAmount;
		return network;
	}

	public static <E extends MutableNeuralNetwork> E shiftWeightsRandom(E network, double maxOffset) throws NullPointerException {
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

	public static <E extends MutableNeuralNetwork> E adjustBias(E network, int layer, int node, double adjustAmount) throws NullPointerException, ArrayIndexOutOfBoundsException {
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

	public static <E extends MutableNeuralNetwork> E crossover(Supplier<E> parent1, Supplier<E> parent2, E child) throws LayoutMismatchException, NullPointerException{
		return crossover(parent1.get(), parent2.get(), child);
	}

	public static <E extends MutableNeuralNetwork> E crossover(E parent1, E parent2, E child) throws LayoutMismatchException, NullPointerException{
		return crossover(parent1, parent2, child, 0.5f);
	}
	

	public static <E extends MutableNeuralNetwork> E crossover(E parent1, E parent2, E child, float lerpFactor) throws LayoutMismatchException, NullPointerException{
		NeuralNetworkTools.requireSameDimensions(parent1, parent2, child);

		double[][][] weights1 = parent1.weights, weights2 = parent2.weights, childWeights = child.weights;

		try {
			for(int layer = 0; layer < weights1.length; layer++)
				for(int node = 0; node < weights1[layer].length; node++){
					double[] w1 = weights1[layer][node], w2 = weights2[layer][node], c = childWeights[layer][node];
					for(int weight = 0; weight < w1.length; weight++)
						c[weight] = lerp(w1[weight], w2[weight], lerpFactor);
				}
		} catch (Exception e) {
			e.printStackTrace();
		}

		return child;
	}

	public static <E extends MutableNeuralNetwork> ArrayList<E> getRandomizedNetworks(int numNetworks, Supplier<E> networkSupplier) throws IllegalArgumentException, NullPointerException{
		Objects.requireNonNull(networkSupplier, "Network Supplier is null.");

		ArrayList<E> networks = new ArrayList<>(numNetworks);
		for(int i = 0; i < numNetworks; i++)
			networks.add(randomizeWeightsAndBiases(networkSupplier.get()));

		return networks;
	}

	public static <E extends MutableNeuralNetwork> ArrayList<E> getRandomizedNetworks(int numNetworks, Supplier<E> networkSupplier, double weightOrigin, double weightBound, double biasOrigin, double biasBound) throws IllegalArgumentException, NullPointerException{
		Objects.requireNonNull(networkSupplier, "Network Supplier is null.");

		ArrayList<E> networks = new ArrayList<>(numNetworks);
		for(int i = 0; i < numNetworks; i++)
			networks.add(randomizeWeightsAndBiases(networkSupplier.get(), weightOrigin, weightBound, biasOrigin, biasBound));

		return networks;
	}

	public static <E extends MutableNeuralNetwork> List<E> getMutations(Supplier<E> copySupplier, int numMutations, double maxWeightDeviation, double maxBiasDeviation, double mutationRate) throws IllegalArgumentException, NullPointerException{
		Objects.requireNonNull(copySupplier, "Network Supplier is null.");
		ArrayList<E> mutations = new ArrayList<>(numMutations);
		
		for(int i = 0; i < numMutations; i++)
			mutations.add(mutate(copySupplier.get(), maxWeightDeviation, maxBiasDeviation, mutationRate));

		return mutations;
	}

	public static <E extends MutableNeuralNetwork> E mutate(E network, double maxWeightDeviation, double maxBiasDeviation, double mutationRate, Random random){
		double[][][] weights = network.weights;
		double[][] biases = network.biases;

		for(int layer = 0; layer < weights.length; layer++)
			for(int node = 0; node < weights[layer].length; node++){
				if(random.nextDouble() < mutationRate) biases[layer][node] = randomOffset(biases[layer][node], maxBiasDeviation, random);
				for(int weight = 0; weight < weights[layer][node].length; weight++)
					if(random.nextDouble() < mutationRate) weights[layer][node][weight] = randomOffset(weights[layer][node][weight], maxWeightDeviation, random);
			}

		return network;
	}

	public static <E extends MutableNeuralNetwork> E mutate(E network, double maxWeightDeviation, double maxBiasDeviation, double mutationRate){
		return mutate(network, maxWeightDeviation, maxBiasDeviation, mutationRate, RandomNumberGeneratorHolder.RANDOM);
	}

	public static <E extends MutableNeuralNetwork> E mutate(E network, double maxWeightDeviation, double maxBiasDeviation){
		return mutate(network, maxWeightDeviation, maxBiasDeviation, DEFAULT_MUTATION_RATE);
	}

	public static <E extends MutableNeuralNetwork> E mutate(E network){
		return mutate(network, DEFAULT_WEIGHT_DEVIATION, DEFAULT_BIAS_DEVIATION);
	}

	public static <E extends MutableNeuralNetwork> Set<E> createUniqueMutations(Supplier<E> copySupplier, int numMutations, double maxWeightDeviation, double maxBiasDeviation) throws IllegalArgumentException, NullPointerException{
		HashSet<E> mutations = new HashSet<>(numMutations);

		while(mutations.size() < numMutations)
			mutations.add(mutate(copySupplier.get(), maxWeightDeviation, maxBiasDeviation));

		return mutations;
	}

	public static <E extends MutableNeuralNetwork> Set<E> createUniqueChildren(E parent1, E parent2, Supplier<E> childSource, int numChildren) throws IllegalArgumentException, NullPointerException{
		HashSet<E> children = new HashSet<>(numChildren);

		while(children.size() < numChildren)
			children.add(NeuralNetworkTools.crossover(parent1, parent2, childSource.get(), (float)Math.random()));

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


	public static <E extends SimpleNeuralNetwork> void requireSameLayout(Collection<E> networks) throws LayoutMismatchException, NullPointerException{
		if ( !haveSameLayout(networks) )
			throw new LayoutMismatchException("Networks have different layouts.");
	}


	public static void requireSameLayout(SimpleNeuralNetwork... networks) throws LayoutMismatchException, NullPointerException{
		if ( !haveSameLayout(networks) )
			throw new LayoutMismatchException("Networks have different layouts.");
	}


	public static void requireSameLayout(SimpleNeuralNetwork network1, SimpleNeuralNetwork network2) throws LayoutMismatchException, NullPointerException{
		if ( !haveSameLayout(network1, network2) )
			throw new LayoutMismatchException("Networks have different layouts.");
	}

	public static boolean haveSameLayout(SimpleNeuralNetwork... networks) throws NullPointerException {
		
		for (int i = 0, next = 1; next < networks.length; i = next++)
			if(!haveSameLayout(networks[i], networks[next])) return false;

		return true;
	}

	public static <E extends SimpleNeuralNetwork> boolean haveSameLayout(Collection<E> networks) throws NullPointerException {
		Objects.requireNonNull(networks, "networks is null");
		if(networks.size() < 2) return true;

		Iterator<E> iterator = networks.iterator();

		for (E a = iterator.next(), b; iterator.hasNext();){
			b = iterator.next();
			if(!haveSameLayout(a, b)) return false;
			a = b;
		}

		return true;
	}

	public static boolean haveSameLayout(SimpleNeuralNetwork network1, SimpleNeuralNetwork network2) throws NullPointerException {		
		return network1.LAYOUT.equals(network2.LAYOUT);
	}

	public static <E extends SimpleNeuralNetwork> boolean haveSameWeights(Collection<E> networks) throws NullPointerException {
		Objects.requireNonNull(networks, "networks is null");
		if(networks.size() < 2) return true;

		Iterator<E> iterator = networks.iterator();

		for (E a = iterator.next(), b; iterator.hasNext();){
			b = iterator.next();
			if(!haveSameWeights(a, b)) return false;
			a = b;
		}

		return true;
	}

	public static boolean haveSameWeights(SimpleNeuralNetwork... networks) throws ArrayIndexOutOfBoundsException {
		
		for (int i = 0, next = 1; next < networks.length; i = next++)
			if(!haveSameWeights(networks[i], networks[next])) return false;

		return true;
	}

	public static boolean haveSameWeights(SimpleNeuralNetwork network1, SimpleNeuralNetwork network2) throws ArrayIndexOutOfBoundsException {
		double[][][] weights1 = network1.weights,
					 weights2 = network2.weights;

		for(int layer = 0; layer < weights1.length; layer++)
			for(int node = 0; node < weights1[layer].length; node++)
				for(int weight = 0; weight < weights1[layer][node].length; weight++)
					if(weights1[layer][node][weight] != weights2[layer][node][weight])
						return false;
		
		return true;
	}

	public static <E extends SimpleNeuralNetwork> boolean haveSameBiases(Collection<E> networks) throws NullPointerException {
		Objects.requireNonNull(networks, "networks is null");
		if(networks.size() < 2) return true;

		Iterator<E> iterator = networks.iterator();

		for (E a = iterator.next(), b; iterator.hasNext();){
			b = iterator.next();
			if(!haveSameBiases(a, b)) return false;
			a = b;
		}

		return true;
	}

	public static boolean haveSameBiases(SimpleNeuralNetwork... networks) throws ArrayIndexOutOfBoundsException {
		
		for (int i = 0, next = 1; next < networks.length; i = next++)
			if(!haveSameBiases(networks[i], networks[next])) return false;

		return true;
	}

	public static boolean haveSameBiases(SimpleNeuralNetwork network1, SimpleNeuralNetwork network2) throws ArrayIndexOutOfBoundsException {		
		double[][] biases1 = network1.biases;
		double[][] biases2 = network2.biases;

		for(int layer = 0; layer < biases1.length; layer++)
			for(int node = 0; node < biases1[layer].length; node++)
				if(biases1[layer][node] != biases2[layer][node])
					return false;

		return true;
	}

	public static void requireSameDimensions(SimpleNeuralNetwork network1, SimpleNeuralNetwork network2) throws DimensionsMismatchException, NullPointerException {
		if(!haveSameDimensions(network1, network2)) throw new DimensionsMismatchException("Network dimensions are not equal.");
	}

	public static void requireSameDimensions(SimpleNeuralNetwork... networks) throws DimensionsMismatchException, NullPointerException {
		if(!haveSameDimensions(networks)) throw new DimensionsMismatchException("Network dimensions are not equal.");
	}

	public static <E extends SimpleNeuralNetwork> void requireSameDimensions(Collection<E> networks) throws DimensionsMismatchException, NullPointerException {
		if(!haveSameDimensions(networks)) throw new DimensionsMismatchException("Network dimensions are not equal.");
	}

	public static <E extends SimpleNeuralNetwork> boolean haveSameDimensions(Collection<E> networks) throws NullPointerException {
		Objects.requireNonNull(networks, "networks is null");
		if(networks.size() < 2) return true;

		Iterator<E> iterator = networks.iterator();

		for (E a = iterator.next(), b; iterator.hasNext();){
			b = iterator.next();
			if(!haveSameDimensions(a, b)) return false;
			a = b;
		}

		return true;
	}

	public static boolean haveSameDimensions(SimpleNeuralNetwork... networks) throws NullPointerException {
		for (int i = 0, next = 1; next < networks.length; i = next++) 
			if(!haveSameDimensions(networks[i], networks[next])) return false;

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

	final protected static double lerp(double min, double max, double frac){
		return min + frac * (max - min);
	}

	final public static double[][] deepCopy(double[][] original){
		double[][] copy = new double[original.length][];
		for(int i = 0; i < original.length; i++){
			copy[i] = new double[original[i].length];
			System.arraycopy(original[i], 0, copy[i], 0, original[i].length);
		}
		return copy;
	}

	final public static double[][][] deepCopy(double[][][] original){
		double[][][] copy = new double[original.length][][];
		for(int i = 0; i < original.length; i++){
			copy[i] = new double[original[i].length][];
			for(int j = 0; j < original[i].length; j++){
				copy[i][j] = new double[original[i][j].length];
				System.arraycopy(original[i][j], 0, copy[i][j], 0, original[i][j].length);
			}
		}
		return copy;
	}

	private static class RandomNumberGeneratorHolder{
		final public static Random RANDOM = new Random();
	}
}

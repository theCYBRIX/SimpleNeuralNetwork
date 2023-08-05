package mjsd.simpleneuralnetwork.training;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Supplier;

import mjsd.simpleneuralnetwork.NetworkLayout;
import mjsd.simpleneuralnetwork.NeuralNetworkTools;

public class EvolutionaryPopulation {

	final static float DEF_ELITE_FRACTION = 0.05f;
	final static float DEF_PARENT_FRACTION = 0.1f;

	final static float DEF_CROSSOVER_FRACTION = 0.3f;
	final static float DEF_MUTATION_FRACTION = 1 - DEF_CROSSOVER_FRACTION;

	final static float DEF_AGGRESSIVE_MUTATION_FRACTION = 0.5f;
	final static float DEF_SUBTLE_MUTATION_FRACTION = 1 - DEF_AGGRESSIVE_MUTATION_FRACTION;

	final private static float HIGH_BIAS_ADJUST_RATE = 1.0f,
                               LOW_BIAS_ADJUST_RATE = 0.002f,
                               HIGH_WEIGHT_ADJUST_RATE = 0.4f,
                               LOW_WEIGHT_ADJUST_RATE = 0.002f;

	final private NetworkLayout NETWORK_LAYOUT;
	final private Supplier<RankedNeuralNetwork> NETWORK_SUPPLIER;
	
	private ArrayList<RankedNeuralNetwork> eliteNetworks, parentNetworks, currentGeneration;
	private LinkedList<RankedNeuralNetwork> manualAdditions = new LinkedList<>();
	private int totalNumNetworks,
				numEliteNetworks,
				numParentNetworks,
				numOffspringNetworks;

	private int numCrossovers,
				totalNumMutations,
				numAggressiveMutations,
				numSubtleMutations;

	private float eliteFraction = DEF_ELITE_FRACTION,
				  parentFraction = DEF_PARENT_FRACTION,
				  crossoverFraction = DEF_CROSSOVER_FRACTION,
				  aggressiveMutationFraction = DEF_AGGRESSIVE_MUTATION_FRACTION;

    private double subtleBiasDelta = LOW_BIAS_ADJUST_RATE,
                   aggressiveBiasDelta = HIGH_BIAS_ADJUST_RATE,
                   subtleWeightDelta = LOW_WEIGHT_ADJUST_RATE,
                   aggressiveWeightDelta = HIGH_WEIGHT_ADJUST_RATE;

	private Optional<Double> bestScore = Optional.empty();

	private int generation;


	public EvolutionaryPopulation(int numNetworks, NetworkLayout layout) throws IllegalArgumentException, NullPointerException {
		if(numNetworks <= 0) throw new IllegalArgumentException("Cannot have less than 1 network in a Population.");

		NETWORK_LAYOUT = Objects.requireNonNull(layout);
		NETWORK_SUPPLIER = () -> new RankedNeuralNetwork(NETWORK_LAYOUT);

		totalNumNetworks = numNetworks;
		updatePopulationDistribution();

		eliteNetworks = new ArrayList<>(numEliteNetworks);
		parentNetworks = new ArrayList<>(numParentNetworks);
		currentGeneration = new ArrayList<>(totalNumNetworks);
	}

	public void reset(){
		generation = 0;
		bestScore = Optional.empty();

		eliteNetworks.clear();
		parentNetworks.clear();
		currentGeneration.clear();
	}

	/**
	 * Checks if the Population contains the exact parameter object, such that {@code parameterObject == objectInList}.
	 * @param o The object to search for.
	 * @return {@code true} if and only if the Population contains the exact object provided in the parameter.
	 */
	public boolean containsExact(Object o){
		for(RankedNeuralNetwork participant : this.currentGeneration)
			if(participant == o) return true;

		return false;
	}

	public RankedNeuralNetwork[] getEliteNetworks() {
		return eliteNetworks.toArray(new RankedNeuralNetwork[eliteNetworks.size()]);
	}

	public int getGeneration() {
		return generation;
	}

	public Optional<Double> getBestScore() {
		return bestScore;
	}
	
	public List<RankedNeuralNetwork> getCurrentGeneration() {
		return currentGeneration;
	}
	


	public void populateNewGeneration(){
		while(parentNetworks.size() < numParentNetworks)
			parentNetworks.add(EvolutionaryTrainer.getRandomizedNetwork(NETWORK_SUPPLIER));

		synchronized(currentGeneration){
			currentGeneration.clear();

			synchronized(manualAdditions){
				while(manualAdditions.size() > 0)
					currentGeneration.add(manualAdditions.removeFirst());
			}

			//Uphold elitism by keeping best performers
			currentGeneration.addAll(eliteNetworks);

			RankedNeuralNetwork mutation;
			Iterator<RankedNeuralNetwork> parentIterator = parentNetworks.iterator();
			for(int i = 0; i < numSubtleMutations; i++){
				//Add subtle mutations of network
				if(!parentIterator.hasNext()) parentIterator = parentNetworks.iterator();
				mutation = parentIterator.next().copy();
				NeuralNetworkTools.shiftWeightsAndBiases(mutation, subtleWeightDelta, subtleBiasDelta);
				currentGeneration.add(mutation);
			}

			for(int i = 0; i < numAggressiveMutations; i++){
				//Add aggressive mutations of network
				if(!parentIterator.hasNext()) parentIterator = parentNetworks.iterator();
				mutation = parentIterator.next().copy();
				NeuralNetworkTools.shiftWeightsAndBiases(mutation, aggressiveWeightDelta, aggressiveBiasDelta);
				currentGeneration.add(mutation);
			}

			if(numCrossovers > 0){
				int parentIndex1 = 0, parentIndex2 = 1;
				RankedNeuralNetwork parent1 = parentNetworks.get(parentIndex1);

				//Add children of best performing networks
				for(int i = 0; i < numCrossovers; i++){
					currentGeneration.add(NeuralNetworkTools.singlePointCrossover(parent1, parentNetworks.get(parentIndex2), NETWORK_SUPPLIER));

					parentIndex2++;
					if(parentIndex2 == parentIndex1) parentIndex2++;
					if(parentIndex2 >= parentNetworks.size()){
						parentIndex1++;
						if(parentIndex1 == parentNetworks.size()){
							parentIndex1 = 0;
							parentIndex2 = 1;
						} else {
							parentIndex2 = 0;
						}
						parent1 = parentNetworks.get(parentIndex1);
					}
				}
			}

			generation++;
		}
	}

	public void add(RankedNeuralNetwork network) throws NullPointerException{
		Objects.requireNonNull(network);

		synchronized(manualAdditions){
			manualAdditions.add(network);
		}
	}

	public void addAll(Collection<RankedNeuralNetwork> networks) throws NullPointerException{
		for(RankedNeuralNetwork network : networks)
			Objects.requireNonNull(network);

		synchronized(manualAdditions){
			manualAdditions.addAll(networks);
		}
	}

	public void resetDistribution(){
		eliteFraction = DEF_ELITE_FRACTION;
		parentFraction = DEF_PARENT_FRACTION;
		crossoverFraction = DEF_CROSSOVER_FRACTION;
		aggressiveMutationFraction = DEF_AGGRESSIVE_MUTATION_FRACTION;
		updatePopulationDistribution();
	}

	protected void processScores(Collection<RankedNeuralNetwork> networks){
		if(networks.isEmpty()){
			return;
		}

		LinkedList<RankedNeuralNetwork> scoreBoard = new LinkedList<>(networks);

		Collections.sort(scoreBoard);
		bestScore = scoreBoard.getLast().getScore();
/*
		StringBuilder readout = new StringBuilder("\nScores:");
		for(RankedNeuralNetwork network : scoreBoard)
			readout.append("\n").append(network.getScore().isPresent() ? network.getScore().get() : "N/A");
		System.out.println(readout.toString());
*/

		while(scoreBoard.size() < numParentNetworks)
			scoreBoard.addFirst(EvolutionaryTrainer.getRandomizedNetwork(NETWORK_SUPPLIER));

		eliteNetworks.clear();
		eliteNetworks.addAll(scoreBoard.subList(scoreBoard.size() - numEliteNetworks, scoreBoard.size()));

		for(int i = 0; i < numParentNetworks; i++)
			parentNetworks.add(scoreBoard.removeLast());
	}

	public void setNumAggressiveMutations(int numAggressiveMutations) throws IllegalArgumentException {
		if(numAggressiveMutations < 0) throw new IllegalArgumentException("Cannot set less than 0 mutations.");
		if(numAggressiveMutations > totalNumMutations) throw new IllegalArgumentException("Cannot set more aggressive mutations than total number of mutations.");
		aggressiveMutationFraction = (numAggressiveMutations / (float)totalNumMutations);
		updateMutationDistribution();
	}

	public void setNumSubtleMutations(int numSubtleMutations) throws IllegalArgumentException {
		if(numSubtleMutations < 0) throw new IllegalArgumentException("Cannot set less than 0 mutations.");
		if(numSubtleMutations > totalNumMutations) throw new IllegalArgumentException("Cannot set more subtle mutations than total number of mutations.");
		aggressiveMutationFraction = 1 - (numSubtleMutations / (float)totalNumMutations);
		updateMutationDistribution();
	}

	public void setNumCrossovers(int numCrossovers) throws IllegalArgumentException {
		if(numCrossovers < 0) throw new IllegalArgumentException("Cannot set less than 0 crossovers.");
		if(numCrossovers > numEliteNetworks) throw new IllegalArgumentException("Cannot set more crossovers than total number of offspring networks.");
		crossoverFraction = numCrossovers / (float)numOffspringNetworks;
		updatePopulationDistribution();
	}

	public void setTotalNumMutations(int numMutations) throws IllegalArgumentException {
		if(numMutations < 0) throw new IllegalArgumentException("Cannot set less than 0 mutations.");
		if(numMutations > numOffspringNetworks) throw new IllegalArgumentException("Cannot set more mutations than total number of offspring networks.");
		crossoverFraction = 1 - (numMutations / (float)numOffspringNetworks);
		updatePopulationDistribution();
	}

	public void setNumEliteNetworks(int numElites) throws IllegalArgumentException {
		if(numElites < 0) throw new IllegalArgumentException("Cannot set less than 0 elite networks.");
		if(numElites > totalNumNetworks) throw new IllegalArgumentException("Cannot set more elite networks than total number of networks.");

		eliteFraction = numElites / (float)totalNumNetworks;
		updatePopulationDistribution();

		eliteNetworks.ensureCapacity(numEliteNetworks);
	}

	public void setNumParentNetworks(int numParents) throws IllegalArgumentException {
		if(numParents <= 0) throw new IllegalArgumentException("Cannot set less than 1 parent network.");
		if(numParents > numOffspringNetworks) throw new IllegalArgumentException("Cannot set more parent networks than total number of offspring networks.");
		
		parentFraction = numParents / (float)totalNumNetworks;
		updatePopulationDistribution();

		parentNetworks.ensureCapacity(numParentNetworks);
	}

	public void setNumNetworks(int numNetworks) throws IllegalArgumentException {
		if(numNetworks <= 0) throw new IllegalArgumentException("Cannot have less than 1 network in a Population.");

		totalNumNetworks = numNetworks;
		updatePopulationDistribution();

		eliteNetworks.ensureCapacity(numEliteNetworks);
		parentNetworks.ensureCapacity(numParentNetworks);
		currentGeneration.ensureCapacity(totalNumNetworks);
	}

	private void updatePopulationDistribution(){
		numEliteNetworks = Math.round(totalNumNetworks * eliteFraction);
		numParentNetworks = Math.max(Math.round(totalNumNetworks * parentFraction), 1);
		numOffspringNetworks = totalNumNetworks - numEliteNetworks;

		numCrossovers = (numParentNetworks > 1) ? Math.round(crossoverFraction * numOffspringNetworks) : 0;
		totalNumMutations = numOffspringNetworks - numCrossovers;
		updateMutationDistribution();
	}

	private void updateMutationDistribution(){
		numAggressiveMutations = Math.round(aggressiveMutationFraction * totalNumMutations);
		numSubtleMutations = totalNumMutations - numAggressiveMutations;
	}


	public void setBiasAggressiveAdjustment(double adjustmentRate) throws IllegalArgumentException{
		this.aggressiveBiasDelta = ensureValidRate(adjustmentRate);
	}

	public void setWeightAggressiveAdjustment(double adjustmentRate) throws IllegalArgumentException{
		this.aggressiveWeightDelta = ensureValidRate(adjustmentRate);
	}

	public void setBiasSubtleAdjustment(double adjustmentRate) throws IllegalArgumentException{
		this.subtleBiasDelta = ensureValidRate(adjustmentRate);
	}

	public void setWeightSubtleAdjustment(double adjustmentRate) throws IllegalArgumentException{
		this.subtleWeightDelta = ensureValidRate(adjustmentRate);
	}

	public void setSubtleAdjustment(double maxWeightDelta, double maxBiasDelta) throws IllegalArgumentException{
        ensureValidRate(maxWeightDelta);
        ensureValidRate(maxBiasDelta);
		this.subtleWeightDelta = maxWeightDelta;
		this.subtleBiasDelta = maxBiasDelta;
	}

	public void setAggressiveAdjustment(double maxWeightDelta, double maxBiasDelta) throws IllegalArgumentException{
        ensureValidRate(maxWeightDelta);
        ensureValidRate(maxBiasDelta);
		this.aggressiveWeightDelta = maxWeightDelta;
		this.aggressiveBiasDelta = maxBiasDelta;
	}

    private double ensureValidRate(double rate) throws IllegalArgumentException{
        if(rate < 0) throw new IllegalArgumentException("Cannot set rate below zero.");
        return rate;
    }

}
package com.mjsd.simpleneuralnetwork.training.evolution;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Supplier;

import com.mjsd.simpleneuralnetwork.NeuralNetworkTools;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.ScoredNetwork;

public class EvolutionaryTrainer<E extends MutableNeuralNetwork, T extends Comparable<T>> implements Runnable {

	final private TrainingScenario<E, T> TRAINING_SCENARIO;
	final private NetworkEvolutionManager<E, T> EVOLUTION_MANAGER;

	private boolean running = false, keepAlive;

	private Thread runningThread = null;

	private LinkedList<Consumer<EvolutionaryTrainer<E, T>>> postGenerationCallbacks = new LinkedList<>();

	private int generation = 1, numNetworks;

	private ArrayList<ScoredNetwork<E, T>> neuralNetworks, previousGeneration = new ArrayList<>();

	private Comparator<ScoredNetwork<E, T>> comparator;

	public EvolutionaryTrainer(int numNetworks, NetworkEvolutionManager<E, T> evolutionManager, TrainingScenario<E, T> trainingScenario) throws IllegalArgumentException, NullPointerException {
		this(numNetworks, evolutionManager, trainingScenario, null);
	}

	public EvolutionaryTrainer(int numNetworks, NetworkEvolutionManager<E, T> evolutionManager, TrainingScenario<E, T> trainingScenario, Comparator<ScoredNetwork<E, T>> comparator) throws IllegalArgumentException, NullPointerException {
		if(numNetworks <= 0)
			throw new IllegalArgumentException("NumNetworks must be >= 1.");
		this.TRAINING_SCENARIO = Objects.requireNonNull(trainingScenario);
		this.EVOLUTION_MANAGER = Objects.requireNonNull(evolutionManager);
		this.comparator = Objects.requireNonNullElse(comparator, (x, y) -> x.compareTo(y));
		this.numNetworks = numNetworks;
		this.neuralNetworks = new ArrayList<>(numNetworks);
	}

	@Override
	public void run() {
		if(running) return;
		try{
			keepAlive = true;
			running = true;
			runningThread = Thread.currentThread();
			Collection<ScoredNetwork<E, T>> newGeneration;
			
			if (neuralNetworks.size() < numNetworks)
				neuralNetworks.addAll(EVOLUTION_MANAGER.createRandomGeneration(numNetworks - neuralNetworks.size()));

			while(keepAlive){

				TRAINING_SCENARIO.setNetworks(neuralNetworks);
				TRAINING_SCENARIO.run();
				TRAINING_SCENARIO.evaluateNetworks();

				for(Consumer<EvolutionaryTrainer<E, T>> callback : postGenerationCallbacks)
					callback.accept(this);

				if(Thread.interrupted()) break;
				
				newGeneration = EVOLUTION_MANAGER.createNewGeneration(neuralNetworks, numNetworks, comparator);
				previousGeneration = neuralNetworks;
				neuralNetworks = new ArrayList<>(newGeneration);

				generation++;
			}
		} catch (Exception e){
			e.printStackTrace();
		} finally {
			running = false;
			keepAlive = false;
			runningThread = null;
		}
	}

	public void attachCallback(Consumer<EvolutionaryTrainer<E, T>> callback){
		postGenerationCallbacks.add(callback);
	}

	public boolean detachCallback(Consumer<EvolutionaryTrainer<E, T>> callback){
		return postGenerationCallbacks.remove(callback);
	}

	public void detachAllCallbacks(){
		postGenerationCallbacks.clear();
	}

	public void addNetwork(E network) throws NullPointerException{
		neuralNetworks.add(new ScoredNetwork<E, T>(network));
	}

	public void addNetwork(ScoredNetwork<E, T> network) throws NullPointerException{
		neuralNetworks.add(network);
	}

	public void addAll(Collection<? extends E> networks) throws NullPointerException{
		networks.forEach(x -> neuralNetworks.add(new ScoredNetwork<E, T>(x)));
	}

	final protected static <T extends MutableNeuralNetwork> T getRandomizedNetwork(Supplier<T> networkSupplier){
		return NeuralNetworkTools.randomizeWeightsAndBiases(networkSupplier.get());
	}

	public void stop() throws SecurityException{
		keepAlive = false;
		if(runningThread != null) runningThread.interrupt();
	}

	public boolean isRunning() {
		return running;
	}

	public Optional<Thread> getRunningThread() {
		return Optional.ofNullable(runningThread);
	}

	public int getGeneration() {
		return generation;
	}

	public void setGeneration(int generation) {
		this.generation = generation;
	}
	
	public List<ScoredNetwork<E, T>> getPreviousGeneration() {
		return previousGeneration;
	}

}

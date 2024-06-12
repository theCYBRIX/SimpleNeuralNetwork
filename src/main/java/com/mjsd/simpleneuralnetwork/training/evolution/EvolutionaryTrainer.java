package com.mjsd.simpleneuralnetwork.training.evolution;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Supplier;

import com.mjsd.simpleneuralnetwork.NeuralNetworkTools;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.RankedNeuralNetwork;

public class EvolutionaryTrainer<E extends RankedNeuralNetwork> implements Runnable {

	final private TrainingScenario<E> TRAINING_SCENARIO;
	final private Population<E> POPULATION;

	private boolean running = false, keepAlive;

	private Thread runningThread = null;

	private ArrayList<Consumer<EvolutionaryTrainer<E>>> postGenerationCallbacks = new ArrayList<>();

	private int generation = 1;

	public EvolutionaryTrainer(Population<E> ecosystem, TrainingScenario<E> trainingScenario) throws IllegalArgumentException, NullPointerException {
		this.TRAINING_SCENARIO = Objects.requireNonNull(trainingScenario);
		this.POPULATION = Objects.requireNonNull(ecosystem);
	}

	@Override
	public void run() {
		if(running) return;
		try{
			keepAlive = true;
			running = true;
			runningThread = Thread.currentThread();
			Collection<E> newGeneration;
			POPULATION.ensureSufficientNetworks();
			while(keepAlive){

				newGeneration = POPULATION.getMembers();

				TRAINING_SCENARIO.setParticipants(newGeneration);
				TRAINING_SCENARIO.run();
				TRAINING_SCENARIO.evaluateParticipants();

				for(Consumer<EvolutionaryTrainer<E>> callback : postGenerationCallbacks)
					callback.accept(this);

				if(Thread.interrupted()) break;

				POPULATION.populateNewGeneration();
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

	public void attachCallback(Consumer<EvolutionaryTrainer<E>> callback){
		postGenerationCallbacks.add(callback);
	}

	public boolean detachCallback(Consumer<EvolutionaryTrainer<E>> callback){
		return postGenerationCallbacks.remove(callback);
	}

	public void detachAllCallbacks(){
		postGenerationCallbacks.clear();
	}

	public void add(E network) throws NullPointerException{
		POPULATION.add(network);
	}

	public void addAll(Collection<? extends E> networks) throws NullPointerException{
		POPULATION.addAll(networks);
	}

	public Optional<Double> getBestScore(){
		return POPULATION.getBestScore();
	}
	
	public List<E> getLeaderBoard(){
		return POPULATION.getLeaderBoard();
	}
	
	public List<E> getLeaderBoard(Comparator<E> comparator){
		return POPULATION.getLeaderBoard(comparator);
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
	
	public Population<E> getPopulation() {
		return POPULATION;
	}

}

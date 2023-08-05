package mjsd.simpleneuralnetwork.training;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Supplier;

import mjsd.simpleneuralnetwork.NeuralNetworkTools;
import mjsd.simpleneuralnetwork.training.evolution.Ecosystem;

public class EvolutionaryTrainer<E extends RankedNeuralNetwork> implements Runnable {

	final private static byte DEF_WEIGHT_RANGE = 1, DEF_BIAS_RANGE = 1;


	final private TrainingScenario<E> TRAINING_SCENARIO;
	final private Ecosystem<E> ECOSYSTEM;

	private boolean running = false, keepAlive;

	private Thread runningThread = null;

	private ArrayList<E> manualAdditions = new ArrayList<>();

	private ArrayList<Consumer<EvolutionaryTrainer<E>>> postGenerationCallbacks = new ArrayList<>();

	private int generation = 0;

	public EvolutionaryTrainer(Ecosystem<E> ecosystem, TrainingScenario<E> trainingScenario) throws IllegalArgumentException, NullPointerException {
		this.TRAINING_SCENARIO = Objects.requireNonNull(trainingScenario);
		this.ECOSYSTEM = Objects.requireNonNull(ecosystem);
	}

	@Override
	public void run() {
		if(running) return;
		try{
			keepAlive = true;
			running = true;
			runningThread = Thread.currentThread();
			ArrayList<E> newGeneration;
			while(keepAlive){

				ECOSYSTEM.populateNewGeneration();
				newGeneration = ECOSYSTEM.getCurrentGeneration();

				if(manualAdditions.size() > 0){
					synchronized(manualAdditions){
						newGeneration.addAll(manualAdditions);
						manualAdditions.clear();
					}
				}

				TRAINING_SCENARIO.setParticipants(newGeneration);
				TRAINING_SCENARIO.run();
				TRAINING_SCENARIO.evaluateParticipants();

				generation++;
					
				for(Consumer<EvolutionaryTrainer<E>> callback : postGenerationCallbacks)
					callback.accept(this);

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
		synchronized(manualAdditions){
			manualAdditions.add(network);
		}
	}

	public void addAll(Collection<? extends E> networks) throws NullPointerException{
		synchronized(manualAdditions){
			manualAdditions.addAll(networks);
		}
	}
	

	final protected static <T extends MutableNeuralNetwork> T getRandomizedNetwork(Supplier<T> networkSupplier){
		T network = networkSupplier.get();

		NeuralNetworkTools.randomizeWeightsAndBiases(network, DEF_WEIGHT_RANGE, DEF_BIAS_RANGE);

		return network;
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
	
	public Ecosystem<E> getEcosystem() {
		return ECOSYSTEM;
	}

}

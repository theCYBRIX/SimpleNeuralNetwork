package com.github.thecybrix.simpleneuralnetwork.training.evolution;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.simpleneuralnetwork.util.AutoRunnable;
import com.github.thecybrix.simpleneuralnetwork.util.CallbackInvoker;

public class EvolutionaryTrainer<E extends MutableNeuralNetwork> implements AutoRunnable, CallbackInvoker<EvolutionaryTrainer<E>> {
	final private static Logger LOGGER = Logger.getLogger(EvolutionaryTrainer.class.getName());

	static {
		LOGGER.setLevel(Level.OFF);
	}

	final private LinkedList<Consumer<EvolutionaryTrainer<E>>> CALLBACKS = new LinkedList<>();

	final protected NetworkEvolutionManager<E> EVOLUTION_MANAGER;

	final private TrainingScenario<E> TRAINING_SCENARIO;
 
	protected int generation, numNetworks;
	protected List<ScoredNetwork<E>> neuralNetworks;
    private Comparator<ScoredNetwork<E>> comparator;


    private List<ScoredNetwork<E>> previousGeneration = Collections.emptyList();

	private boolean running = false, keepAlive;
	private Thread runningThread = null;

	public EvolutionaryTrainer(int numNetworks, NetworkEvolutionManager<E> evolutionManager, TrainingScenario<E> trainingScenario) throws IllegalArgumentException, NullPointerException {
		this(numNetworks, evolutionManager, trainingScenario, null);
	}

	public EvolutionaryTrainer(int numNetworks, NetworkEvolutionManager<E> evolutionManager, TrainingScenario<E> trainingScenario, Comparator<ScoredNetwork<E>> comparator) throws IllegalArgumentException, NullPointerException {
		if(numNetworks <= 0)
			throw new IllegalArgumentException("NumNetworks must be >= 1.");
		this.EVOLUTION_MANAGER = Objects.requireNonNull(evolutionManager);
		this.comparator = Objects.requireNonNullElse(comparator, (x, y) -> x.compareTo(y));
		this.numNetworks = numNetworks;
		this.neuralNetworks = new ArrayList<>(numNetworks);
		this.TRAINING_SCENARIO = Objects.requireNonNull(trainingScenario);
	}

	@Override
	public void run() {
		if(running) return;
		try{
			keepAlive = true;
			running = true;
			runningThread = Thread.currentThread();
			
			if (neuralNetworks.size() < numNetworks)
				neuralNetworks.addAll(EVOLUTION_MANAGER.createRandomGeneration(numNetworks - neuralNetworks.size()));

			while(keepAlive){

				TRAINING_SCENARIO.execute(neuralNetworks);

				if(Thread.interrupted()) break;
				
				previousGeneration = List.copyOf(neuralNetworks);

				neuralNetworks = EVOLUTION_MANAGER.createNewGeneration(neuralNetworks, numNetworks);

				processCallbacks(this);

				generation++;
			}
		} catch (InterruptedException e){
			if(keepAlive)
				LOGGER.log(Level.WARNING, "EvolutionaryTrainer was interrupted unexpectedly.", e);;
		} catch (Exception e){
			LOGGER.log(Level.SEVERE, e.getMessage(), e);
		} finally {
			running = false;
			keepAlive = false;
			runningThread = null;
		}
	}

	public void addNetwork(E network) throws NullPointerException{
		neuralNetworks.add(new ScoredNetwork<E>(network));
	}

	public void addNetwork(ScoredNetwork<E> network) throws NullPointerException{
		neuralNetworks.add(network);
	}

	public void addAll(Collection<? extends E> networks) throws NullPointerException{
		networks.forEach(x -> neuralNetworks.add(new ScoredNetwork<E>(x)));
	}

	public void addAllScored(Collection<? extends ScoredNetwork<E>> networks) throws NullPointerException{
		neuralNetworks.addAll(networks);
	}
    
	public int getGeneration() {
		return generation;
	}

	public void setGeneration(int generation) {
		this.generation = generation;
	}

	public Comparator<ScoredNetwork<E>> getComparator() {
        return comparator;
    }

    public void setComparator(Comparator<ScoredNetwork<E>> comparator) {
        this.comparator = comparator;
    }

	public List<ScoredNetwork<E>> getPreviousGeneration() {
		return previousGeneration;
	}

	public List<ScoredNetwork<E>> getNetworks() {
		return Collections.unmodifiableList(neuralNetworks);
	}

	public NetworkEvolutionManager<E> getEvolutionManager() {
		return EVOLUTION_MANAGER;
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

	@Override
	public List<Consumer<EvolutionaryTrainer<E>>> getCallbackList() {
		return CALLBACKS;
	}
}
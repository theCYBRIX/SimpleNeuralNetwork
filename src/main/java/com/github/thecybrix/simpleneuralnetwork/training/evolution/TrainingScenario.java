package com.github.thecybrix.simpleneuralnetwork.training.evolution;

import java.util.Collection;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;

@FunctionalInterface
public interface TrainingScenario<E extends MutableNeuralNetwork> {
	/**
	 * Executes the training scenario on the given collection of scored networks.
	 * <p>
	 * Evaluates the performance of each network within the collection and updates
	 * their scores accordingly. This method guarantees that upon normal completion, every
	 * {@code ScoredNetwork} in the collection will have its score updated to reflect its
	 * performance in this scenario.
	 * </p>
	 *
	 * @param networks the collection of {@code ScoredNetwork} objects to be evaluated
	 * @throws InterruptedException if any thread has interrupted the current thread.
	 * @see {@link ScoredNetwork}
	 */
	public void execute(Collection<ScoredNetwork<E>> networks) throws InterruptedException;
}
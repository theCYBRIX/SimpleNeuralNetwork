package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.Collection;

import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.ScoredNetwork;

public interface TrainingScenario<E extends MutableNeuralNetwork, T extends Comparable<T>> extends Runnable {
	public void setNetworks(Collection<ScoredNetwork<E, T>> c);
	public void evaluateNetworks();
}
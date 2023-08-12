package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.Collection;

import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;

public interface TrainingScenario<E extends MutableNeuralNetwork> extends Runnable {
	public void setParticipants(Collection<E> c);
	public void evaluateParticipants();
}
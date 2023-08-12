package com.mjsd.simpleneuralnetwork.training;

import java.util.Optional;

import com.mjsd.simpleneuralnetwork.NetworkLayout;
import com.mjsd.simpleneuralnetwork.NeuralNetworkBuilder;
import com.mjsd.simpleneuralnetwork.NeuralNetworkTools;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork;

public class RankedNeuralNetwork extends MutableNeuralNetwork implements Comparable<RankedNeuralNetwork> {

	private Optional<Double> score = Optional.empty();

	public RankedNeuralNetwork(NetworkLayout layout) throws NullPointerException {
		super(layout);
	}

	public RankedNeuralNetwork(SimpleNeuralNetwork template) throws NullPointerException {
		super(template);
	}

	public RankedNeuralNetwork(NetworkLayout layout, OutputHandler outputHandler, InputProvider inputProvider) throws NullPointerException {
		super(layout, outputHandler, inputProvider);
	}

	protected RankedNeuralNetwork(NetworkLayout layout, double[][][] weights, double[][] biases, OutputHandler outputHandler, InputProvider inputProvider) throws NullPointerException {
		super(layout, weights, biases, outputHandler, inputProvider);
	}

	public Optional<Double> getScore() {
		return score;
	}

	public void setScore(Double score) {
		this.score = Optional.ofNullable(score);
	}

	@Override
	public int compareTo(RankedNeuralNetwork o) {
		if(this.score.isPresent()){
			return (o.score.isPresent()) ? this.score.get().compareTo(o.score.get()) : 1;
		} else {
			return (o.score.isPresent()) ? -1 : 0;
		}
	}

	@Override
	public RankedNeuralNetwork copy() {
		return new RankedNeuralNetwork(LAYOUT, NeuralNetworkTools.deepCopy(weights), NeuralNetworkTools.deepCopy(biases), this.getOutputHandler(), this.getInputProvider());
	}

	@Override
	public NeuralNetworkBuilder<? extends RankedNeuralNetwork> newBuilder(){
		return new RankedNeuralNetworkBuilder(this);
	}
    
}

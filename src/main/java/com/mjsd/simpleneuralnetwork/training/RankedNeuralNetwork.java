package com.mjsd.simpleneuralnetwork.training;

import java.util.Optional;

import com.google.gson.annotations.JsonAdapter;
import com.mjsd.simpleneuralnetwork.NetworkLayout;
import com.mjsd.simpleneuralnetwork.NeuralNetworkBuilder;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork;
import com.mjsd.simpleneuralnetwork.gson.RankedNeuralNetworkAdapter;

@JsonAdapter(RankedNeuralNetworkAdapter.class)
public class RankedNeuralNetwork extends MutableNeuralNetwork implements Comparable<RankedNeuralNetwork> {

	private Optional<Double> score = Optional.empty();

	public RankedNeuralNetwork(NetworkLayout layout) throws NullPointerException {
		super(layout);
	}

	public RankedNeuralNetwork(SimpleNeuralNetwork template) throws NullPointerException {
		super(template);
	}

	protected RankedNeuralNetwork(NetworkLayout layout, double[][][] weights, double[][] biases, OutputHandler[] outputHandlers, InputProvider[] inputProviders) throws NullPointerException {
		super(layout, weights, biases, outputHandlers, inputProviders);
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
		return new RankedNeuralNetwork(this);
	}

	@Override
	public NeuralNetworkBuilder<? extends RankedNeuralNetwork> newBuilder(){
		return new RankedNeuralNetworkBuilder(this);
	}
    
}

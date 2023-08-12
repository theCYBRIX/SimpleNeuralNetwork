package com.mjsd.simpleneuralnetwork.training;

import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork;

public interface RewardFunction {
	public double getReward(SimpleNeuralNetwork network);
}
package mjsd.simpleneuralnetwork.training;

import mjsd.simpleneuralnetwork.SimpleNeuralNetwork;

public interface RewardFunction {
	public double getReward(SimpleNeuralNetwork network);
}
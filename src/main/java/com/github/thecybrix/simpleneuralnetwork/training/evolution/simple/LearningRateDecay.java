package com.github.thecybrix.simpleneuralnetwork.training.evolution.simple;

import java.util.function.Consumer;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.EvolutionaryTrainer;

public class LearningRateDecay<E extends MutableNeuralNetwork> implements Consumer<EvolutionaryTrainer<E>> {
    final private static float DEFAULT_DECAY_FACTOR = 0.99f;
    
    private float decayFactor;

    public LearningRateDecay(){
        decayFactor = DEFAULT_DECAY_FACTOR;
    }

    public LearningRateDecay(float decayFactor){
        this.decayFactor = decayFactor;
    }

    @Override
    public void accept(EvolutionaryTrainer<E> trainer) {
        trainer.setLearningRate(decay(trainer.getInitialLearningRate(), decayFactor, trainer.getGeneration()));
    }

    public float getDecayFactor() {
        return decayFactor;
    }

    public void setDecayFactor(float decayFactor) {
        this.decayFactor = decayFactor;
    }

    final public static double decay(double learningRate, double decayFactor, int generation){
        return learningRate * Math.pow(decayFactor, generation);
    }
}

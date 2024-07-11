package com.mjsd.simpleneuralnetwork.training;

import java.util.Objects;
import java.util.Optional;
import java.util.function.Supplier;

import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork;

public class ScoredNetwork<E extends SimpleNeuralNetwork, T extends Comparable<T>> implements Comparable<ScoredNetwork<E, T>>, Supplier<E> {
    private E network; 
    private Optional<T> score;

    public ScoredNetwork(E network) {
        this(network, Optional.empty());
    }

    public ScoredNetwork(E network, Class<T> c) {
        this(network, Optional.empty());
    }

    public ScoredNetwork(E network, Optional<T> score) {
        this.network = Objects.requireNonNull(network, "Network is null.");
        this.score = Objects.requireNonNull(score, "Score is null. Use ScoredNetwork(network) to initialize to Optional.empty() automatically.");
    }

    @Override
    public E get() {
        return network;
    }
    

    public void set(E network) {
        this.network = Objects.requireNonNull(network, "Network is null.");
    }

    public Optional<T> getScore() {
        return score;
    }

    public void setScore(T score) {
        this.score = Optional.ofNullable(score);
    }

    @Override
    public int compareTo(ScoredNetwork<E, T> other) {
        Optional<T> otherScore = other.getScore();
        if(score.isPresent())
            return otherScore.isPresent() ? score.get().compareTo(otherScore.get()) : 1;
        else
            return otherScore.isPresent() ? -1 : 0;
    }
}

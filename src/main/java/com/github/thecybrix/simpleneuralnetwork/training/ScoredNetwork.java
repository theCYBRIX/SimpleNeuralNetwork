package com.github.thecybrix.simpleneuralnetwork.training;

import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.OptionalDouble;
import java.util.function.Supplier;

import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;

public class ScoredNetwork<E extends SimpleNeuralNetwork> implements Comparable<ScoredNetwork<E>>, Supplier<E> {
    
    private E network; 
    private OptionalDouble score = OptionalDouble.empty();

    private ScoredNetwork(){}

    public ScoredNetwork(E network) {
        this.network = Objects.requireNonNull(network, "Network is null.");
    }

    public ScoredNetwork(E network, double score) {
        this(network);
        setScore(score);
    }

    public ScoredNetwork(E network, OptionalDouble score) {
        this(network);
        setScore(score);
    }

    public ScoredNetwork(ScoredNetwork<E> other) {
        this(Objects.requireNonNull(other, "other is null.").network);
        setScore(other.score);
    }

    @Override
    public E get() {
        return network;
    }
    

    public void set(E network) {
        this.network = Objects.requireNonNull(network, "Network is null.");
    }

    public OptionalDouble getScore() {
        return score;
    }

    public void setScore(double score) {
        this.score = OptionalDouble.of(score);
    }

    public void setScore(OptionalDouble score) throws NullPointerException {
        this.score = Objects.requireNonNull(score, "score (OptionalDouble) is null");
    }

    public void clearScore(){
        this.score = OptionalDouble.empty();
    }

    @Override
    public boolean equals(Object obj) {
        if(obj == this) return true;
        if( !(obj instanceof ScoredNetwork) ) return false;
        
        ScoredNetwork<?> other = (ScoredNetwork<?>) obj;
        return (score.isEmpty() && other.score.isEmpty()) || (Double.compare(score.getAsDouble(), other.score.getAsDouble()) == 0);
    }

    @Override
    public int compareTo(ScoredNetwork<E> other) {
        OptionalDouble otherScore = other.getScore();
        if(score.isPresent())
            return otherScore.isPresent() ? Double.compare(score.getAsDouble(), otherScore.getAsDouble()) : 1;
        else
            return otherScore.isPresent() ? -1 : 0;
    }

    public static <E extends SimpleNeuralNetwork> ScoredNetwork<E> empty(){
        return new ScoreHolder<>();
    }

    private static class ScoreHolder<E extends SimpleNeuralNetwork> extends ScoredNetwork<E>{
        @Override
        public E get() {
            throw new NoSuchElementException("Unble to get network from a ScoreHolder.");
        }
        @Override
        public void set(E network) {
            throw new UnsupportedOperationException("Unable to set network on a ScoreHolder.");
        }
    }
}

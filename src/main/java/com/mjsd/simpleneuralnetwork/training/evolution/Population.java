package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

import com.mjsd.simpleneuralnetwork.exceptions.DimensionsMismatchException;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;

public interface Population<E extends  MutableNeuralNetwork> extends Parallelizable {
    public void ensureSufficientNetworks();
    public void populateNewGeneration();
    public int size();
    public void setSize(int numNetworks) throws IllegalArgumentException;
    public Collection<E> getMembers();
    public void add(E network) throws DimensionsMismatchException, NullPointerException;
    public void addAll(Collection<? extends E> networks) throws DimensionsMismatchException, NullPointerException;
	public Optional<Double> getBestScore();
	public List<E> getLeaderBoard();
	public List<E> getLeaderBoard(Comparator<E> comparator) throws NullPointerException;
}

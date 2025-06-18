package com.github.thecybrix.simpleneuralnetwork.util;

import java.util.NoSuchElementException;

public class Pair<E, T> {
    final private E first;
    final private T second;

    public Pair(E first, T second) {
        this.first = first;
        this.second = second;
    }

    public static <E, T> Pair<E, T> of(E first, T second){
        return new Pair<>(first, second);
    }

    public E getFirst() {
        return first;
    }

    public T getSecond() {
        return second;
    }

    public E getFirstOrElse(E other) {
        return (first != null) ? first : other;
    }

    public T getSecondOrElse(T other) {
        return (second != null) ? second : other;
    }

    public E getFirstOrThrow() throws NoSuchElementException {
        if(first == null) throw new NoSuchElementException("First object is null.");
        return first;
    }

    public T getSecondOrThrow() throws NoSuchElementException {
        if(second == null) throw new NoSuchElementException("Second object is null.");
        return second;
    }

    public boolean hasFirst(){
        return first != null;
    }

    public boolean hasSecond(){
        return second != null;
    }

    public Pair<E, T> copy(){
        return new Pair<>(first, second);
    }
}

package com.github.thecybrix.simpleneuralnetwork.util;

import java.util.NoSuchElementException;

public class Pair<E, T> {
    final private E FIRST;
    final private T SECOND;

    public Pair(E first, T second) {
        FIRST = first;
        SECOND = second;
    }

    public static <E, T> Pair<E, T> of(E first, T second){
        return new Pair<>(first, second);
    }

    public E getFirst() {
        return FIRST;
    }

    public T getSecond() {
        return SECOND;
    }

    public E getFirstOrElse(E other) {
        return (FIRST != null) ? FIRST : other;
    }

    public T getSecondOrElse(T other) {
        return (SECOND != null) ? SECOND : other;
    }

    public E getFirstOrThrow() throws NoSuchElementException {
        if(FIRST == null) throw new NoSuchElementException("First object is null.");
        return FIRST;
    }

    public T getSecondOrThrow() throws NoSuchElementException {
        if(SECOND == null) throw new NoSuchElementException("Second object is null.");
        return SECOND;
    }

    public boolean hasFirst(){
        return FIRST != null;
    }

    public boolean hasSecond(){
        return SECOND != null;
    }
}

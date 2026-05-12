package com.github.thecybrix.simpleneuralnetwork.util;

import java.util.NoSuchElementException;
import java.util.OptionalInt;

public class intObjPair<T> {
    final T obj;
    final OptionalInt integer;

    public intObjPair(T object, int integer){
        this.obj = object;
        this.integer = OptionalInt.of(integer);
    }

    public static <E> intObjPair<E> of(E object, int integer){
        return new intObjPair<>(object, integer);
    }

    public T getObject() {
        return obj;
    }

    public int getInteger() throws NoSuchElementException {
        return integer.getAsInt();
    }

    public T getObjectOrElse(T other) {
        return obj != null ? obj : other;
    }

    public int getIntegerOrElse(int other) throws NoSuchElementException {
        return integer.isPresent() ? integer.getAsInt() : other;
    }

    public T getObjectOrThrow(){
        if(obj == null) throw new NoSuchElementException();
        return obj;
    }

    public int getIntegerOrThrow(){
        if(integer.isEmpty()) throw new NoSuchElementException();
        return integer.getAsInt();
    }

    public boolean hasObject() {
        return obj != null;
    }

    public boolean hasInteger() {
        return integer.isPresent();
    }

    public intObjPair<T> copy(){
        return new intObjPair<>(obj, integer.getAsInt());
    }
}

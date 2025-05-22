package com.github.thecybrix.simpleneuralnetwork.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;

import com.github.thecybrix.simpleneuralnetwork.exceptions.ArraySizeMismatchException;

public abstract class WeightedRandomSelector {
    
    public static <E, T extends Number> List<E> select(int numItems, List<E> objects, Function<E, T> weightFunction) throws ArraySizeMismatchException, IllegalArgumentException, NullPointerException {
        return select(numItems, objects, weightFunction, false, null);
    }
    
    public static <E, T extends Number> List<E> select(int numItems, List<E> objects, Function<E, T> weightFunction, boolean invert) throws ArraySizeMismatchException, IllegalArgumentException, NullPointerException {
        return select(numItems, objects, weightFunction, invert, null);
    }
    
    public static <E, T extends Number> List<E> select(int numItems, List<E> objects, Function<E, T> weightFunction, boolean invert, Random random) throws ArraySizeMismatchException, IllegalArgumentException, NullPointerException {
        return select(numItems, objects, objects.stream().map(weightFunction).collect(Collectors.toList()), invert, random);
    }
    
    public static <E, T extends Number> List<E> select(int numItems, Map<E, T> objects, boolean invert) throws ArraySizeMismatchException, IllegalArgumentException, NullPointerException {
        return select(numItems, mapToList(objects), invert);
    }

    public static <E, T extends Number> List<E> select(int numItems, List<E> objects, List<T> weights) throws ArraySizeMismatchException, IllegalArgumentException, NullPointerException {
        return select(numItems, objects, weights, null);
    }

    public static <E, T extends Number> List<E> select(int numItems, List<E> objects, List<T> weights, Random random) throws ArraySizeMismatchException, IllegalArgumentException, NullPointerException {
        return select(numItems, objects, weights, false, random);
    }

    public static <E, T extends Number> List<E> select(int numItems, List<E> objects, List<T> weights, boolean invert, Random random) throws ArraySizeMismatchException, IllegalArgumentException, NullPointerException {
        if(numItems <= 0) throw new IllegalArgumentException("Cannot select negative or zero items. (numItems <= 0)");
        if(numItems > objects.size()) throw new IllegalArgumentException("Cannot select more items than there are in total.");

        Objects.requireNonNull(objects, "Objects list is null.");
        Objects.requireNonNull(weights, "Weights list is null.");

        if(objects.size() != weights.size()) throw new ArraySizeMismatchException("Object and weight lists have differing sizes. (" + objects.size() + " != " + weights.size() +")");
        if(objects.parallelStream().anyMatch(x -> x == null)) throw new NullPointerException("A weight may not be null.");

        if(random == null) random = new Random();

        ArrayList<WeightedObject<E>> weightedObjects = new ArrayList<>(objects.size());
        Iterator<E> objectIterator = objects.iterator();
        Iterator<? extends Number> weightIterator = weights.iterator();
        for (int i = 0; i < objects.size(); i++)
            weightedObjects.add(new WeightedObject<E>(objectIterator.next(), weightIterator.next()));
        
        return select(numItems, weightedObjects, invert, random);
    }

    public static <E> List<E> select(int numItems, List<WeightedObject<E>> objects) {
        return select(numItems, objects, false);
    }

    public static <E> List<E> select(int numItems, List<WeightedObject<E>> objects, boolean invert) {
        return select(numItems, objects, invert, new Random());
    }

    public static <E> List<E> select(int numItems, List<WeightedObject<E>> objects, boolean invert, Random random) {
        if(numItems <= 0) throw new IllegalArgumentException("Cannot select less than zero items.");

        Collections.sort(objects, WeightedObject.COMPARATOR);

        if(invert){
            int high = objects.size() - 1, low = 0;
            while(high > low){
                WeightedObject<E> h, l;
                h = objects.get(high);
                l = objects.get(low);
                double temp = h.weight;
                h.weight = l.weight;
                l.weight = temp;
            }
            Collections.reverse(objects);
        }

        Iterator<WeightedObject<E>> objIterator = objects.iterator();
        double cumulativeWeight = objIterator.next().weight;
        while(objIterator.hasNext()){
            WeightedObject<E> objA = objIterator.next();
            objA.weight += cumulativeWeight;
            cumulativeWeight = objA.weight;
        }

        ArrayList<E> out = new ArrayList<>(numItems);
        
        for(int i = 0; i < numItems; i++){

            int index = Collections.binarySearch(objects, random.nextDouble() * cumulativeWeight);
            if(index < 0) index = -(index + 1);

            WeightedObject<E> selected = objects.remove(index);
            out.add(selected.OBJECT);


            if(index > 0) selected.weight -= objects.get(index - 1).weight;
            cumulativeWeight -= selected.weight;
            for (WeightedObject<E> obj : objects.subList(index, objects.size())) {
                obj.weight -= selected.weight;
            }
        }

        return out;
    }

    private static <E, T extends Number> List<WeightedObject<E>> mapToList(Map<E, T> map) throws NullPointerException {
        ArrayList<WeightedObject<E>> list = new ArrayList<>(map.size());
        for(Entry<E, T> entry : map.entrySet())
            list.add(new WeightedObject<E>(entry.getKey(), entry.getValue()));
        return list;
    }

    public static class WeightedObject<E> implements Comparable<Double>{
        final public static Comparator<WeightedObject<?>> COMPARATOR = (x, y) -> Double.compare(x.weight, y.weight);
        final private E OBJECT;
        private double weight;

        public WeightedObject(E oBJ, Number weight) {
            OBJECT = oBJ;
            setWeight(weight.doubleValue());
        }

        public E getObject() {
            return OBJECT;
        }

        public Double getWeight() {
            return weight;
        }

        public void setWeight(Double weight) {
            this.weight = Objects.requireNonNull(weight, "Weight is null.");
        }

        @Override
        public int compareTo(Double o) {
            return Double.compare(weight, o);
        }
    } 
}
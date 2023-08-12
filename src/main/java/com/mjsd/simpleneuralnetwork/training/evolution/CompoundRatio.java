package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Objects;
import java.util.stream.Stream;

import org.apache.commons.numbers.fraction.Fraction;

import com.mjsd.simpleneuralnetwork.exceptions.IllegalTermCountException;

public class CompoundRatio implements Iterable<Fraction>{
    final public static String TERM_DELIMITER = ":";
    final public static CompoundRatio UNIFORM_DISTRIBUTION = new CompoundRatio(1);

    final private Fraction[] FRACTIONS;

    private CompoundRatio(String ratio) throws NumberFormatException, IllegalArgumentException, NullPointerException{
        this(stringToRatioTerms(ratio));
    }

    private CompoundRatio(int... terms) throws IllegalArgumentException, NullPointerException{
        if(Objects.requireNonNull(terms, "Terms array is null.").length == 0)
            throw new IllegalArgumentException("Illegal number of terms. (terms.length == 0)");
        if(Arrays.stream(terms).anyMatch(x -> x < 0))
            throw new IllegalArgumentException("A term may not be less than zero.\nTerms = " + Arrays.toString(terms));

        FRACTIONS = new Fraction[terms.length];

        int sum = terms[0];
        for(int i = 1; i < terms.length; i++)
            sum += terms[i];

        for(int i = 0; i < terms.length; i++)
            FRACTIONS[i] = Fraction.of(terms[i], sum);
    }


    public static CompoundRatio uniform(){
        return UNIFORM_DISTRIBUTION;
    }


    public static CompoundRatio uniform(int numTerms) throws IllegalArgumentException{
        if(numTerms <= 0) throw new IllegalArgumentException("Illegal number of terms. (numTerms <= 0)");

        int[] terms = new int[numTerms];
        for(int i = 0; i < terms.length; i++)
            terms[i] = 1;

        return new CompoundRatio(terms);
    }

    public static CompoundRatio of(int... terms) throws IllegalArgumentException, NullPointerException{
        return new CompoundRatio(terms);
    }

    public static CompoundRatio fromString(String terms) throws NumberFormatException, IllegalArgumentException, NullPointerException{
        return new CompoundRatio(terms);
    }
    


    public int getNumTerms(){
        return FRACTIONS.length;
    }

    public Fraction getFraction(int index) throws ArrayIndexOutOfBoundsException{
        return FRACTIONS[index];
    }

    public Stream<Fraction> stream(){
        return Arrays.stream(FRACTIONS);
    }

    @Override
    public Iterator<Fraction> iterator() {
        return new Iterator<Fraction>() {
            int index = 0;

            @Override
            public boolean hasNext() {
                return index < FRACTIONS.length;
            }

            @Override
            public Fraction next() {
                return FRACTIONS[index++];
            }
        };
    }


    public static CompoundRatio requireNumberOfTerms(CompoundRatio ratio, int numTerms) throws IllegalTermCountException, NullPointerException{
        return requireNumberOfTerms(ratio, numTerms, "Invalid number of terms. Required " + numTerms + " but found " + ratio.getNumTerms());
    }

    public static CompoundRatio requireNumberOfTerms(CompoundRatio ratio, int numTerms, String exceptionMessage) throws IllegalTermCountException, NullPointerException{
        if (ratio.getNumTerms() != numTerms)
            throw new IllegalTermCountException(exceptionMessage);
        
        return ratio;
    }

    private static int[] stringToRatioTerms(String ratio) throws NumberFormatException, NullPointerException {
        String[] termsStr = ratio.split(TERM_DELIMITER);
        int[] terms = new int[termsStr.length];
        for(int i = 0; i < termsStr.length; i++)
            terms[i] = Integer.parseInt(termsStr[i]);
        return terms;
    }
}

package com.github.thecybrix.simpleneuralnetwork.util;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Objects;
import java.util.stream.Stream;

import com.github.thecybrix.simpleneuralnetwork.exceptions.IllegalTermCountException;

public class MultiPartRatio implements Iterable<Fraction>{
    final public static String TERM_DELIMITER = ":";
    final public static MultiPartRatio UNIFORM_ONE_TERM = new MultiPartRatio(1);

    final private Fraction[] FRACTIONS;

    private MultiPartRatio(String ratio) throws NumberFormatException, IllegalArgumentException, NullPointerException{
        this(stringToRatioTerms(ratio));
    }

    private MultiPartRatio(int... terms) throws IllegalArgumentException, NullPointerException{
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

    /**
     * @return a ratio where the first and only term is 1.
     * @apiNote Equivalent to {@code uniform(1)}
     * @see {@link MultiPartRatio#uniform(int)}
     */
    public static MultiPartRatio uniform(){
        return UNIFORM_ONE_TERM;
    }

    public static MultiPartRatio uniform(int numTerms) throws IllegalArgumentException{
        if(numTerms <= 0) throw new IllegalArgumentException("Illegal number of terms. (numTerms <= 0)");
        if(numTerms == 1) return UNIFORM_ONE_TERM;

        int[] terms = new int[numTerms];
        Arrays.fill(terms, 1);

        return new MultiPartRatio(terms);
    }


    /**
     * Creates a MultiPartRatio where a single term has all the weight (value of 1),
     * and all other terms have a weight of 0.
     * <p>
     * This is effectively a "one-hot" distribution, useful for cases where
     * one category or partition receives 100% of the proportion.
     *
     * @param index the index of the term that should receive all the weight (must be between 0 and totalTerms - 1)
     * @param totalTerms the total number of terms in the ratio (must be >= 1)
     * @return a MultiPartRatio with one term set to 1 and all others set to 0
     * @throws IllegalArgumentException if totalTerms is less than 1,
     *                                  or if index is out of bounds
     */
    public static MultiPartRatio singleton(int index, int totalTerms) throws IllegalArgumentException, IndexOutOfBoundsException {
        if(totalTerms <= 0) throw new IllegalArgumentException("Total terms must be >= 1.");
        if(index < 0 || index >= totalTerms) throw new IndexOutOfBoundsException(String.format("Index %d out of range for MultiPartRatio of size %d.", index, totalTerms));

        int[] terms = new int[totalTerms];
        terms[index] = 1;

        return new MultiPartRatio(terms);
    }

    public static MultiPartRatio of(int... terms) throws IllegalArgumentException, NullPointerException{
        return new MultiPartRatio(terms);
    }

    public static MultiPartRatio fromString(String terms) throws NumberFormatException, IllegalArgumentException, NullPointerException{
        return new MultiPartRatio(terms);
    }
    


    public int getNumTerms(){
        return FRACTIONS.length;
    }

    public Fraction getFraction(int index) throws ArrayIndexOutOfBoundsException{
        return FRACTIONS[index];
    }

    /**
     * Distributes a given total amount across the terms of this ratio, returning an array of integers
     * representing the allocated portions for each term.
     * <p>
     * The distribution is performed proportionally using the Largest Remainder Method:
     * <ul>
     *   <li>The total is first multiplied by each term's fractional share (defined as term value / sum of all terms).</li>
     *   <li>The integer floor of each result is assigned initially.</li>
     *   <li>If the sum of initial assignments is less than the total, the remaining amount (at most 1)
     *       is assigned to the term with the largest remainder.</li>
     * </ul>
     * This ensures that the sum of the returned array equals the input {@code total}, and that the
     * distribution is as close as possible to the intended proportions.
     *
     * @param total the total amount to distribute (must be non-negative)
     * @return an array of integers representing the distributed amounts per term,
     *         where the length of the array equals the number of terms in this ratio
     * @throws IllegalArgumentException if {@code total} is negative
     */
    public int[] distribute(int total){
        if(total <= 0){
            if(total == 0)
                return new int[getNumTerms()];
            else
                throw new IllegalArgumentException("Total cannot be < 0"); 
        }
        
        int[] allocations = new int[getNumTerms()];
        float[] remainders = new float[allocations.length];
        int assignedTotal = 0;
        for(int i = 0; i < allocations.length; i++){
            float idealCount = total * getFraction(i).floatValue();
            int assignedCount = (int)idealCount;
            allocations[i] = assignedCount;
            assignedTotal += assignedCount;
            remainders[i] = idealCount - assignedCount;
        }

        if(assignedTotal != total){
            int maxIndex = 0;
            for (int i = 1; i < remainders.length; i++) {
                if(remainders[i] > remainders[maxIndex])
                    maxIndex = i;
            }
            allocations[maxIndex] += (total - assignedTotal);
        }

        return allocations;
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

    @Override
    public String toString() {
        return toString(TERM_DELIMITER);
    }

    private String toString(String delimiter){
        StringBuilder builder = new StringBuilder(FRACTIONS[0].getNumerator());

        for(Fraction f : FRACTIONS){
            builder.append(delimiter).append(f.getNumerator());
        }

        return builder.toString();
    }

    public static MultiPartRatio requireNumberOfTerms(MultiPartRatio ratio, int numTerms) throws IllegalTermCountException, NullPointerException{
        return requireNumberOfTerms(ratio, numTerms, "Invalid number of terms. Required " + numTerms + " but found " + ratio.getNumTerms());
    }

    public static MultiPartRatio requireNumberOfTerms(MultiPartRatio ratio, int numTerms, String exceptionMessage) throws IllegalTermCountException, NullPointerException{
        if (ratio.getNumTerms() != numTerms)
            throw new IllegalTermCountException(exceptionMessage);
        
        return ratio;
    }

    private static int[] stringToRatioTerms(String ratio) throws NumberFormatException, NullPointerException {
        return stringToRatioTerms(ratio, TERM_DELIMITER);
    }

    private static int[] stringToRatioTerms(String ratio, String delimiter) throws NumberFormatException, NullPointerException {
        String[] termsStr = ratio.split(delimiter);
        int[] terms = new int[termsStr.length];
        for(int i = 0; i < termsStr.length; i++)
            terms[i] = Integer.parseInt(termsStr[i]);
        return terms;
    }
}

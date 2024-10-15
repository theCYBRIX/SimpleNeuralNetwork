package com.github.thecybrix.util;

public class Fraction extends Number implements Comparable<Fraction> {
    
    final public static Fraction ONE, ZERO;

    static {
        ONE = new Fraction(1, 1){
            @Override
            public float floatValue() {
                return 1;
            }

            @Override
            public double doubleValue() {
                return 1;
            }

            @Override
            public int intValue() {
                return 1;
            }

            @Override
            public long longValue() {
                return 1;
            }
        };

        ZERO = new Fraction(0, 1){
            @Override
            public float floatValue() {
                return 0;
            }

            @Override
            public double doubleValue() {
                return 0;
            }

            @Override
            public int intValue() {
                return 0;
            }

            @Override
            public long longValue() {
                return 0;
            }
        };
    }
    
    final private int NUMERATOR, DENOMINATOR;

    private Fraction(int numerator, int denominator) {
        NUMERATOR = numerator;
        DENOMINATOR = denominator;
    }

    public static Fraction of(int numerator, int denominator){
        return new Fraction(numerator, denominator);
    }

    @Override
    public float floatValue(){
        return ((float) NUMERATOR) / ((float) DENOMINATOR);
    }

    @Override
    public double doubleValue(){
        return ((double) NUMERATOR) / ((double) DENOMINATOR);
    }

    @Override
    public int intValue() {
        return (int) floatValue();
    }

    @Override
    public long longValue() {
        return (long) doubleValue();
    }

    public int getNumerator() {
        return NUMERATOR;
    }

    public int getDenominator() {
        return DENOMINATOR;
    }

    @Override
    public int compareTo(Fraction other) {
        long thisCrossProduct = (long) this.NUMERATOR * other.DENOMINATOR;
        long otherCrossProduct = (long) other.NUMERATOR * this.DENOMINATOR;
        return Long.compare(thisCrossProduct, otherCrossProduct);
    }

    @Override
    public boolean equals(Object obj) {
        if(obj == this)
            return true;

        if(obj instanceof Fraction)
            return this.compareTo((Fraction) obj) == 0;

        if( !(obj instanceof Number) )
            return false;
        if(obj instanceof Integer)
            return this.intValue() == ((Integer)obj).intValue();
        if(obj instanceof Long)
            return this.longValue() == ((Long)obj).longValue();
        if(obj instanceof Float)
            return Float.compare(this.floatValue(), ((Float)obj).floatValue()) == 0;
        if(obj instanceof Double)
            return Double.compare(this.doubleValue(), ((Double)obj).doubleValue()) == 0;
        
        return false;
    }

    @Override
    public String toString() {
        return NUMERATOR + "/" + DENOMINATOR;
    }
    
}

package com.mjsd.simpleneuralnetwork;

import java.util.Arrays;

public abstract class Normalization {
    
    final public static void Z_Score(double[] vector) {
        double mean = Arrays.stream(vector).sum() / vector.length;
        double variance = Arrays.stream(vector).map(x -> Math.pow(x - mean, 2)).sum() / vector.length;
        double standardDeviation = Math.sqrt(variance);
        for (int i = 0; i < vector.length; i++) {
            vector[i] = (vector[i] - mean) / standardDeviation;
        }
    }
}

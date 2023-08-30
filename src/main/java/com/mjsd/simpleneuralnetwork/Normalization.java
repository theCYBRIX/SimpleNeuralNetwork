package com.mjsd.simpleneuralnetwork;

import java.util.Arrays;

final public class Normalization {

    private Normalization(){}
    
    final public static void Z_Score(double[] vector) {
        double mean = Arrays.stream(vector)
                            .sum() / vector.length;

        double variance = Arrays.stream(vector)
                                .map(x -> Math.pow(x - mean, 2))
                                .sum() / vector.length;

        double standardDeviation = Math.sqrt(variance);

        for (int i = 0; i < vector.length; i++)
            vector[i] = (vector[i] - mean) / standardDeviation;
    }

    final public static void minMaxFeatureScaling(double[] vector){
        if(vector.length == 0) return;

        double min = Arrays.stream(vector).min().getAsDouble(),
               max = Arrays.stream(vector).max().getAsDouble(),
               range = max - min;

        for(int i = 0; i < vector.length; i++)
            vector[i] = (vector[i] - min) / range;
    }

    final public static void minMaxFeatureScaling(double[] vector, double rangeStart, double rangeEnd){
        if(vector.length == 0) return;

        double min = Arrays.stream(vector).min().getAsDouble(),
               max = Arrays.stream(vector).max().getAsDouble(),
               valueRange = max - min,
               range = rangeEnd - rangeStart;

        for(int i = 0; i < vector.length; i++)
            vector[i] = rangeStart + (((vector[i] - min) * range) / valueRange);
    }
}

package com.mjsd.simpleneuralnetwork;

import java.util.Arrays;
import java.util.HashMap;

import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.ActivationFunction;

public enum ActivationFunctions implements ActivationFunction{
	/**
     * LINEAR
	 * <p>
     * Outputs the input values unchanged.
     * </p>
	 */
	LINEAR {
        @Override
        public double apply(double[] layer, int index) {
            return layer[index];
        }
    },
    /**
     * ReLU (Rectified Linear Unit)
     * <p>
     * Sets all negative values to zero, and returns all non-negative values unchanged.
     * </p> 
     */
    ReLU {
        @Override
        public double apply(double[] layer, int index) {
            return Math.max(0, layer[index]);
        }
    },
    /**
     * SIGMOID
     * <p>
     * A logistic function with a distinct S-shaped curve.
     * </p> 
     */
    SIGMOID {
        /**
        * @return 1 / (1 + e^(-x))
        */
        @Override
        public double apply(double[] layer, int index) {
			return (1 / (1 + Math.exp(-layer[index])));
        }
    },
    /**
     * TANH (Hyperbolic Tangent)
     * <p>
     * The Tanh activation function is a hyperbolic tangent sigmoid function that has a range of -1 to 1.
     * </p>
     */
    TANH {
        /**
         * @return (e^x – e^-x) / (e^x + e^-x)
         */
        @Override
        public double apply(double[] layer, int index) {
            double value = layer[index];
            return (Math.exp(value) - Math.exp(-value)) / (Math.exp(value) + Math.exp(-value));
        }
    };

    /**
     * @implNote Softmax caches the values for a given array, and does not check if the array contents change over time.
     * <p>
     * Values must be updated by calling the {@link #updateValues()} method, or by passing a different array to the {@link #apply()} method.
     * </p>
     */
    final public static class Softmax implements ActivationFunction {
        final public static String FUNCTION_NAME = "SOFTMAX";
        private HashMap<Double, Double> valueMap;
        private double[] preppedLayer = null;

        @Override
        public double apply(double[] layer, int index) {
            if(preppedLayer != layer) updateValues(layer);
            return valueMap.get(layer[index]); 
        }

        public void updateValues(double[] layer){
            valueMap = new HashMap<>(layer.length);

            double[] modifiedLayer = new double[layer.length];
            
            double largestInput = Arrays.stream(layer).max().getAsDouble();

            for (int i = 0; i < layer.length; i++)
                modifiedLayer[i] = Math.exp(layer[i] - largestInput);

            Normalization.Z_Score(modifiedLayer);

            for(int i = 0; i < layer.length; i++)
                valueMap.put(layer[i], modifiedLayer[i]);

            preppedLayer = layer;
        }

        @Override
        public String toString() {
            return FUNCTION_NAME;
        }

        @Override
        public boolean equals(Object obj) {
            return (obj != null) && (obj instanceof Softmax);
        }
    }
}

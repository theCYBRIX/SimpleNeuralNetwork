package mjsd.simpleneuralnetwork;

import java.util.Arrays;
import java.util.HashMap;

import mjsd.simpleneuralnetwork.SimpleNeuralNetwork.ActivationFunction;

public enum ActivationFunctions implements ActivationFunction{
	LINEAR {
        @Override
        public double apply(double[] layer, int index) {
            return layer[index];
        }
    },
    ReLU {
        @Override
        public double apply(double[] layer, int index) {
            return Math.max(0, layer[index]);
        }
    },
    SIGMOID {
        @Override
        public double apply(double[] layer, int index) {
			return (1 / (1 + Math.exp(-layer[index])));
        }
    },
    TANH {
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

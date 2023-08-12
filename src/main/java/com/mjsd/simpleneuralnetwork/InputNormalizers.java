package com.mjsd.simpleneuralnetwork;

import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.InputNormalizer;

public enum InputNormalizers implements InputNormalizer{

    NO_NORMALIZER { @Override public void normalize(double[] values) {} },

    BATCH_NORMALIZER {
        @Override
        public void normalize(double[] values) {
			Normalization.Z_Score(values);
        }
    };
    
}

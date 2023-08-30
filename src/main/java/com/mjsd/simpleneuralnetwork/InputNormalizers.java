package com.mjsd.simpleneuralnetwork;

import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.InputNormalizer;

public enum InputNormalizers implements InputNormalizer{

    NONE { @Override public void normalize(double[] values) {} },

    BATCH {
        @Override
        public void normalize(double[] values) {
			Normalization.Z_Score(values);
        }
    },
    
    MIN_MAX {
        @Override
        public void normalize(double[] values) {
			Normalization.minMaxFeatureScaling(values);
        }
    };

    public static class MinMaxNormalization implements InputNormalizer{
        private double rangeStart,
                       rangeEnd;

        public MinMaxNormalization(double rangeStart, double rangeEnd){
            this.rangeStart = rangeStart;
            this.rangeEnd = rangeEnd;
        }

        @Override
        public void normalize(double[] values) {
			Normalization.minMaxFeatureScaling(values, rangeStart, rangeEnd);
        }

        public void setRangeStart(double rangeStart) {
            this.rangeStart = rangeStart;
        }

        public void setRangeEnd(double rangeEnd) {
            this.rangeEnd = rangeEnd;
        }
    }
    
}

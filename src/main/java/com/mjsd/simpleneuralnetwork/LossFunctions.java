package com.mjsd.simpleneuralnetwork;

public final class LossFunctions {

    public static double meanSquaredError(double[] predictedValues, double[] trueValues) throws ArrayIndexOutOfBoundsException, NullPointerException{
        double loss = 0;

        for (int i = 0; i < trueValues.length; i++)
            loss += Math.pow(predictedValues[i] - trueValues[i], 2);

        loss /= trueValues.length;

        return loss;
    }
    
    
	public static double categoricalCrossEntropy(double[] predictedValues, double[] trueValues) throws ArrayIndexOutOfBoundsException, NullPointerException {
        double loss = 0;

        for (int i = 0; i < trueValues.length; i++)
            loss += trueValues[i] * Math.log(predictedValues[i]);

        loss = (-loss);

        return loss;
    }
    
}

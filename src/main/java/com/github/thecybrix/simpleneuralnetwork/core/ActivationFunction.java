package com.github.thecybrix.simpleneuralnetwork.core;

import com.github.thecybrix.simpleneuralnetwork.serialization.json.ActivationFunctionAdapter;
import com.google.gson.annotations.JsonAdapter;

@JsonAdapter(ActivationFunctionAdapter.class)
@FunctionalInterface
public interface ActivationFunction{
	public double apply(double[] layer, int index);

	public default double[] applyAll(double[] layer){
		double[] applied = new double[layer.length];
		
		applyAll(layer, applied);

		return applied;
	}

	public default void applyAll(double[] layer, double[] destination){
		for(int i = 0; i < layer.length; i++)
			destination[i] = apply(layer, i);
	}

	/**
	 * @return This object if the function is context independent, otherwise a copy of this object. 
	 */
	public default ActivationFunction copyOrReuse(){
		return this;
	}

    public static ActivationFunction ensureFunction(ActivationFunction function){
        return (function == null) ? ActivationFunctions.LINEAR : function;
    }
}
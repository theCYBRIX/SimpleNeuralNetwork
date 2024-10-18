package com.github.thecybrix.simpleneuralnetwork.core;

import com.github.thecybrix.simpleneuralnetwork.serialization.json.InputNormalizerAdapter;
import com.google.gson.annotations.JsonAdapter;

@JsonAdapter(InputNormalizerAdapter.class)
@FunctionalInterface
public interface InputNormalizer {
	public void normalize(double[] values);

	/**
	 * @return This object if the normalizer is context independent, otherwise a copy of this object. 
	 */
	public default InputNormalizer copyOrReuse(){
		return this;
	}

    public static InputNormalizer ensureNormalizer(InputNormalizer normalizer){
        return (normalizer == null) ? InputNormalizers.NONE : normalizer;
    }
}
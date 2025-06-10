package com.github.thecybrix.simpleneuralnetwork.serialization.binary;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Map.Entry;

import com.github.thecybrix.simpleneuralnetwork.core.ActivationFunction;
import com.github.thecybrix.simpleneuralnetwork.core.ActivationFunctions;
import com.github.thecybrix.simpleneuralnetwork.core.InputNormalizer;
import com.github.thecybrix.simpleneuralnetwork.core.InputNormalizers;

public abstract class BinaryStorageManager {
    final protected static String FILE_EXTENSION = ".snn";

    final protected static Map<ActivationFunction, String> ACTIVATION_FUNCTION_NAMES = Map.of(ActivationFunctions.LINEAR, "linear", ActivationFunctions.ReLU, "ReLU", ActivationFunctions.SIGMOID, "Sigmoid", ActivationFunctions.TANH, "TanH");
    final protected static Map<InputNormalizer, String> INPUT_NORMALIZER_NAMES = Map.of(InputNormalizers.NONE, "None", InputNormalizers.BATCH, "batch", InputNormalizers.MIN_MAX, "MinMax");
    
    
    protected BinaryStorageManager(){}
    
    protected static <K, V> Map<K,V> invertMap(Map<V,K> map) throws NullPointerException{
        Map<K, V> invertedMap = new HashMap<>(map.size());

        for(Entry<V,K> entry : map.entrySet())
            invertedMap.put(entry.getValue(), entry.getKey());

        return invertedMap;
    }

    protected static <K, V> Map<K, V> requireNonNull(Map<K, V> map) throws NullPointerException{
        return Objects.requireNonNull(map, "Map is null.");
    }

    protected static String requireNonNull(String functionName) throws NullPointerException{
        return Objects.requireNonNull(functionName, "Name cannot be null.");
    }

    protected static InputNormalizer requireNonNull(InputNormalizer normalizer) throws NullPointerException{
        return Objects.requireNonNull(normalizer, "Normalizer cannot be null.");
    }

    protected static ActivationFunction requireNonNull(ActivationFunction function) throws NullPointerException{
        return Objects.requireNonNull(function, "Function cannot be null.");
    }
}

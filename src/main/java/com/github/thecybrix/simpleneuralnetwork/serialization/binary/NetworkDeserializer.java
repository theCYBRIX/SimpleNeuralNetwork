package com.github.thecybrix.simpleneuralnetwork.serialization.binary;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;
import java.util.Objects;

import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkBuilder;
import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork.ActivationFunction;
import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork.InputNormalizer;

public class NetworkDeserializer extends BinaryStorageManager {
    private Map<String, ActivationFunction> activationFunctionNames = BinaryStorageManager.invertMap(BinaryStorageManager.ACTIVATION_FUNCTION_NAMES);
    private Map<String, InputNormalizer> inputNormalizerNames = BinaryStorageManager.invertMap(BinaryStorageManager.INPUT_NORMALIZER_NAMES);

    public NetworkDeserializer(){}

    public NetworkDeserializer(Map<String, ActivationFunction> activationNames, Map<String, InputNormalizer> normalizerNames){
        registerActivationFunctions(activationNames);
        registerInputNormalizers(normalizerNames);
    }

    public <E extends SimpleNeuralNetwork> NeuralNetworkBuilder<E> load(String filePath, NeuralNetworkBuilder<E> networkBuilder) throws FileNotFoundException, IOException, SecurityException, NullPointerException{
        Objects.requireNonNull(networkBuilder, "NetworkBuilder is null.");
        Objects.requireNonNull(filePath, "File path is null.");
        
        File saveFile = new File(filePath);
        if(!saveFile.exists()) throw new FileNotFoundException(filePath);
        
        try(DataInputStream fileReader = new DataInputStream(new FileInputStream(saveFile))){
            double[][][] weights;
            double[][] biases;

            ActivationFunction[] activationFunctions;
            InputNormalizer[] normalizationFunctions;

            //layer count
            int layerCount = fileReader.readInt();
            int layersMinusOne = layerCount - 1;

            weights = new double[layersMinusOne][][];
            biases = new double[layersMinusOne][];
            activationFunctions = new ActivationFunction[layerCount];
            normalizationFunctions = new InputNormalizer[layerCount];
            
            //activation functions
            for (int i = 0; i < activationFunctions.length; i++) {
                activationFunctions[i] = activationFunctionNames.get(fileReader.readUTF());
            }
            
            //normalization functions
            for (int i = 0; i < normalizationFunctions.length; i++) {
                normalizationFunctions[i] = inputNormalizerNames.get(fileReader.readUTF());
            }

            //layer sizes
            int[] layerSizes = new int[layerCount];
            for (int i = 0; i < layerCount; i++) {
                layerSizes[i] = fileReader.readInt();
            }
            
            for (int layer = 1, prevLayer = 0; layer < layerCount; layer++) {
                weights[prevLayer] = new double[layerSizes[layer]][layerSizes[prevLayer]];
                biases[prevLayer] = new double[layerSizes[layer]];
                prevLayer = layer;
            }

            //weights
            for(int layer = 0; layer < weights.length; layer++) {
                for(int node = 0; node < weights[layer].length; node++) {
                    for(int weight = 0; weight < weights[layer][node].length; weight++){
                        weights[layer][node][weight] = fileReader.readDouble();
                    }
                }
            }

            //biases
            for(int layer = 0; layer < biases.length; layer++) {
                for(int node = 0; node < biases[layer].length; node++) {
                    biases[layer][node] = fileReader.readDouble();
                }
            }

            networkBuilder.reset();

            networkBuilder.withInputLayer(layerSizes[0], normalizationFunctions[0], activationFunctions[0]);
            networkBuilder.withOutputLayer(layerSizes[layerSizes.length - 1], normalizationFunctions[normalizationFunctions.length - 1], activationFunctions[activationFunctions.length - 1]);
            for (int i = 1; i < layersMinusOne; i++) {
                networkBuilder.addHiddenLayer(layerSizes[i], normalizationFunctions[i], activationFunctions[i]);
            }
            networkBuilder.withWeights(weights);
            networkBuilder.withBiases(biases);

            return networkBuilder;
        }
        
    }

    public void registerActivationFunctions(Map<String, ActivationFunction> bindings) throws NullPointerException{
        requireNonNull(bindings).forEach((x, y) -> activationFunctionNames.put(x, y));
    }

    public void registerActivationFunction(ActivationFunction function, String name){
        activationFunctionNames.put(requireNonNull(name), requireNonNull(function));
    }

    public void registerInputNormalizers(Map<String, InputNormalizer> bindings){
       requireNonNull(bindings).forEach((x, y) -> inputNormalizerNames.put(x, y));
    }

    public void registerInputNormalizer(InputNormalizer normalizer, String name){
        inputNormalizerNames.put(requireNonNull(name), requireNonNull(normalizer));
    }
}

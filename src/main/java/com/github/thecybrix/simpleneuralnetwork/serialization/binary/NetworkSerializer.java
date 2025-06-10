package com.github.thecybrix.simpleneuralnetwork.serialization.binary;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import com.github.thecybrix.simpleneuralnetwork.core.ActivationFunction;
import com.github.thecybrix.simpleneuralnetwork.core.InputNormalizer;
import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayout;
import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayout.NetworkLayer;

public class NetworkSerializer extends BinaryStorageManager {
    private Map<ActivationFunction, String> activationFunctionNames = new HashMap<>(ACTIVATION_FUNCTION_NAMES);
    private Map<InputNormalizer, String> inputNormalizerNames = new HashMap<>(INPUT_NORMALIZER_NAMES);

    public NetworkSerializer(){}

    public NetworkSerializer(Map<ActivationFunction, String> activationNames, Map<InputNormalizer, String> normalizerNames){
        registerActivationFunctions(activationNames);
        registerInputNormalizers(normalizerNames);
    }

    public void save(SimpleNeuralNetwork network, String filePath, boolean overwrite) throws FileAlreadyExistsException, IOException, SecurityException, NullPointerException{
        Objects.requireNonNull(network, "Network is null.");
        Objects.requireNonNull(filePath, "File path is null.");

        if(!filePath.endsWith(FILE_EXTENSION)) filePath += FILE_EXTENSION;

        File saveFile = new File(filePath);
        if(saveFile.exists()){
            if(!overwrite) throw new FileAlreadyExistsException(filePath);
            if(!saveFile.delete()) throw new IOException("Unable to overwrite existing file. (" + filePath + ")");
            
        } else if(!saveFile.createNewFile()) {
            throw new IOException("Unable to create file. (" + filePath + ")");
        }

        
        try {
            try(DataOutputStream fileWriter = new DataOutputStream(new FileOutputStream(saveFile, false))){
                double[][][] weights = network.getWeights();
                double[][] biases = network.getBiases();
                NetworkLayout layout = NetworkLayout.of(network);
                ArrayList<NetworkLayer> networkLayers = new ArrayList<>(layout.getHiddenLayers().size() + 2);

                networkLayers.add(layout.getInputLayer());
                networkLayers.addAll(layout.getHiddenLayers());
                networkLayers.add(layout.getOutputLayer());

                //layer count
                fileWriter.writeInt(networkLayers.size());

                //activation functions
                for (NetworkLayer layer : networkLayers) {
                    String functionName = activationFunctionNames.get(layer.getActivationFunction());
                    fileWriter.writeUTF(Objects.requireNonNull(functionName, "No name provided for ActivationFunction. (" + layer.getActivationFunction() + ")"));
                }

                //normalization functions
                for (NetworkLayer layer : networkLayers) {
                    String functionName = inputNormalizerNames.get(layer.getInputNormalizer());
                    fileWriter.writeUTF(Objects.requireNonNull(functionName, "No name provided for InputNormalizer. (" + layer.getInputNormalizer() + ")"));
                }

                //layer sizes
                for (NetworkLayer layer : networkLayers) {
                    fileWriter.writeInt(layer.getNodeCount());
                }

                //weights
                for(int layer = 0; layer < weights.length; layer++) {
                    for(int node = 0; node < weights[layer].length; node++) {
                        for(int weight = 0; weight < weights[layer][node].length; weight++){
                            fileWriter.writeDouble(weights[layer][node][weight]);
                        }
                    }
                }

                //biases
                for(int layer = 0; layer < biases.length; layer++) {
                    for(int node = 0; node < biases[layer].length; node++) {
                        fileWriter.writeDouble(biases[layer][node]);
                    }
                }
                
            }
        } catch(Exception e){
            try {
                Files.delete(saveFile.toPath());
            } catch (Exception deletionFailed) {
                e.addSuppressed(new IOException("Unable to delete created file. (" + saveFile.getPath() + ")", deletionFailed));
            }
            
            throw e;
        }
        
    }

    public void registerActivationFunctions(Map<ActivationFunction, String> bindings) throws NullPointerException{
        requireNonNull(bindings).forEach((x, y) -> activationFunctionNames.put(x, y));
    }

    public void registerActivationFunction(ActivationFunction function, String name){
        activationFunctionNames.put(requireNonNull(function), requireNonNull(name));
    }

    public void registerInputNormalizers(Map<InputNormalizer, String> bindings){
        requireNonNull(bindings).forEach((x, y) -> inputNormalizerNames.put(x, y));
    }

    public void registerInputNormalizer(InputNormalizer normalizer, String name){
        inputNormalizerNames.put(requireNonNull(normalizer), requireNonNull(name));
    }
}

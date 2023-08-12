package com.mjsd.simpleneuralnetwork;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import com.google.gson.annotations.JsonAdapter;

import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.ActivationFunction;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.InputNormalizer;
import com.mjsd.simpleneuralnetwork.gson.NetworkLayerAdapter;
import com.mjsd.simpleneuralnetwork.gson.NetworkLayoutAdapter;

@JsonAdapter(NetworkLayoutAdapter.class)
public class NetworkLayout {
    final protected NetworkLayer INPUT, OUTPUT;
    final protected List<NetworkLayer> HIDDEN_LAYERS;


    public NetworkLayout(NetworkLayer input, NetworkLayer output, Collection<NetworkLayer> hiddenLayers) throws NullPointerException {
        this(input, output, Objects.requireNonNull(hiddenLayers).toArray(new NetworkLayer[hiddenLayers.size()]));
    }

    public NetworkLayout(NetworkLayer input, NetworkLayer output, NetworkLayer[] hiddenLayers) throws NullPointerException{
        INPUT = Objects.requireNonNull(input);
        OUTPUT = Objects.requireNonNull(output);

        for(NetworkLayer layer : Objects.requireNonNull(hiddenLayers))
            Objects.requireNonNull(layer);

        HIDDEN_LAYERS = Collections.unmodifiableList(Arrays.asList(hiddenLayers));
    }


    /**
     * @return The input layer of this object, or {@code null} if none has been set.
     */
    public NetworkLayer getInputLayer() {
        return INPUT;
    }


    /**
     * @return The output layer of this object, or {@code null} if none has been set.
     */
    public NetworkLayer getOutputLayer() {
        return OUTPUT;
    }


    /**
     * @return An array containing the hidden layers of this object.
     */
    public List<NetworkLayer> getHiddenLayers() {
        return HIDDEN_LAYERS;
    }

    @Override
    public boolean equals(Object obj) {
        if(! (obj instanceof NetworkLayout)) return false;
        if(this == obj) return true;

        NetworkLayout other = (NetworkLayout)obj;
        
        NetworkLayer thisLayer, otherLayer;
        List<NetworkLayer> thisHiddenLayers = this.getHiddenLayers(),
                           otherHiddenLayers = other.getHiddenLayers();

        if(thisHiddenLayers.size() != otherHiddenLayers.size()) return false;

        for(int i = 0; i < thisHiddenLayers.size(); i++){
            thisLayer = thisHiddenLayers.get(i);
            otherLayer = otherHiddenLayers.get(i);
            if(!thisLayer.equals(otherLayer)) return false;
        }

        thisLayer = this.getInputLayer();
        otherLayer = other.getInputLayer();

        if(!thisLayer.equals(otherLayer)) return false;

        thisLayer = this.getOutputLayer();
        otherLayer = other.getOutputLayer();

        if(!thisLayer.equals(otherLayer)) return false;
        
        return true;
    }

    @Override
    public String toString() {
        StringBuilder out = new StringBuilder();

        out.append(INPUT);

        for (NetworkLayer layer : HIDDEN_LAYERS) {
            out.append("\n").append(layer);
        }

        out.append("\n").append(OUTPUT);

        return out.toString();
    }


    @JsonAdapter(NetworkLayerAdapter.class)
    public static class NetworkLayer {
        final private int NODE_COUNT;
        final private ActivationFunction ACTIVATION_FUNCTION;
        final private InputNormalizer INPUT_NORMALIZER;

        public NetworkLayer(int nodes) throws IllegalArgumentException {
            this(nodes, null, null);
        }

        public NetworkLayer(int nodes, InputNormalizer inputNormalizer) throws IllegalArgumentException {
            this(nodes, inputNormalizer, null);
        }


        public NetworkLayer(int nodes, ActivationFunction activationFunction) throws IllegalArgumentException {
            this(nodes, null, activationFunction);
        }

        public NetworkLayer(int nodes, InputNormalizer inputNormalizer, ActivationFunction activationFunction) throws IllegalArgumentException {
            NODE_COUNT = validateNodeCount(nodes);
            this.ACTIVATION_FUNCTION = ActivationFunction.ensureFunction(activationFunction);
            this.INPUT_NORMALIZER = InputNormalizer.ensureNormalizer(inputNormalizer);
        }

        private int validateNodeCount(int nodeCount) throws IllegalArgumentException {
            if (nodeCount <= 0) throw new IllegalArgumentException("Cannot create a Layer with less than 1 node.");
            return nodeCount;
        }

        public int getNodeCount(){
            return NODE_COUNT;
        }

        public ActivationFunction getActivationFunction(){
            return ACTIVATION_FUNCTION;
        }

        public InputNormalizer getInputNormalizer(){
            return INPUT_NORMALIZER;
        }

        public NetworkLayer copy(){
            return new NetworkLayer(NODE_COUNT, INPUT_NORMALIZER, ACTIVATION_FUNCTION);
        }

        @Override
        public boolean equals(Object obj) {
            if(!(obj instanceof NetworkLayer)) return false;

            NetworkLayer other = (NetworkLayer)obj;

            return (this.getNodeCount() == other.getNodeCount()) && (this.getActivationFunction().equals(other.getActivationFunction())) && (this.getInputNormalizer().equals(other.getInputNormalizer()));
        }

        @Override
        public String toString() {
            StringBuilder out = new StringBuilder();

            out.append("[ 0");
            for (int i = 1; i < NODE_COUNT; i++)
                out.append(", ").append(i);
            out.append(" ]");

            return out.toString();
        }
    }
    
}

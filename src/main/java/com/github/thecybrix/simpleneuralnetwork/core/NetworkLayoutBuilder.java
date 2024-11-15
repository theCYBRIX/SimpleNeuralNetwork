package com.github.thecybrix.simpleneuralnetwork.core;

import java.util.LinkedList;
import java.util.Objects;

import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayout.NetworkLayer;

public class NetworkLayoutBuilder {

    protected NetworkLayer input = null, output = null;
    protected LinkedList<NetworkLayer> hiddenLayers = new LinkedList<>();
    private NetworkLayout layout;

    private boolean layoutCached = false;


    public NetworkLayoutBuilder(){}

    public NetworkLayoutBuilder(int inputs, int outputs) throws IllegalArgumentException {
        this(new NetworkLayer(inputs), new NetworkLayer(outputs));
    }

    public NetworkLayoutBuilder(int inputs, int outputs, ActivationFunction activationFunction) throws IllegalArgumentException {
        this(inputs, activationFunction, outputs, activationFunction);
    }

    public NetworkLayoutBuilder(int inputs, ActivationFunction inputFunction, int outputs, ActivationFunction outputFunction) throws IllegalArgumentException {
        this.input = new NetworkLayer(inputs, inputFunction);
        this.output = new NetworkLayer(outputs, outputFunction);
    }

    public NetworkLayoutBuilder(NetworkLayer inputs, NetworkLayer outputs) {
        this.input = inputs;
        this.output = outputs;
    }

    public NetworkLayoutBuilder(NetworkLayout initialState) throws NullPointerException {
        this.initializeState(initialState);
    }

    public NetworkLayoutBuilder(NetworkLayoutBuilder initialState) throws NullPointerException {
        Objects.requireNonNull(initialState, "initialState is null.");
        this.input = initialState.input.copy();
        this.output = initialState.output.copy();
        this.hiddenLayers = new LinkedList<>(initialState.hiddenLayers);
        if(initialState.layoutCached){
            this.layout = initialState.layout;
            this.layoutCached = true;
        }
    }

    public NetworkLayoutBuilder(SimpleNeuralNetwork initialState) throws NullPointerException {
        Objects.requireNonNull(initialState, "initialState is null.");

        withInputLayer(initialState.inputs.length, initialState.inputNormalizer, initialState.inputActivation);
        withOutputLayer(initialState.outputs.length, initialState.outputNormalizer, initialState.outputActivation);
        for (int layer = 0; layer < initialState.hiddenLayers.length; layer++)
            addLayer(initialState.hiddenLayers[layer].length, initialState.hiddenNormalizers[layer], initialState.hiddenActivations[layer]);
    }



    public NetworkLayout build() throws IllegalStateException{
        if(layoutCached) return layout;

        if(input == null) throw new IllegalStateException("Input layer properties are not specified.");
        if(output == null) throw new IllegalStateException("Output layer properties are not specified.");

        layout = new NetworkLayout(input, output, hiddenLayers);
        layoutCached = true;

        return layout;
    }


    public NetworkLayoutBuilder setState(NetworkLayout layout) throws NullPointerException{
        Objects.requireNonNull(layout, "layout is null.");
        hiddenLayers.clear();

        initializeState(layout);

        return this;
    }

    private void initializeState(NetworkLayout layout){
		input = layout.getInputLayer();
		output = layout.getOutputLayer();
		hiddenLayers.addAll(layout.getHiddenLayers());

        this.layout = layout;
        layoutCached = true;
    }

    public NetworkLayoutBuilder setState(SimpleNeuralNetwork network) throws NullPointerException{
        Objects.requireNonNull(network, "network is null.");
		reset();

        initializeState(network);

        return this;
    }

    private void initializeState(SimpleNeuralNetwork network) throws NullPointerException{
        input = new NetworkLayer(network.inputs.length, network.inputNormalizer, network.inputActivation);
        output = new NetworkLayer(network.outputs.length, network.outputNormalizer, network.outputActivation);
        for (int layer = 0; layer < network.hiddenLayers.length; layer++)
            hiddenLayers.add(new NetworkLayer(network.hiddenLayers[layer].length, network.hiddenNormalizers[layer], network.hiddenActivations[layer]));
    }

    public void reset(){
        this.input = null;
        this.output = null;
        hiddenLayers.clear();

        layoutCached = false;
        layout = null;
    }

    public NetworkLayoutBuilder copy(){
        return new NetworkLayoutBuilder(this);
    }


    private NetworkLayoutBuilder withInputLayerUnchecked(NetworkLayer input) {
        this.input = input;
        layoutCached = false;
        return this;
    }

    public NetworkLayoutBuilder withInputLayer(NetworkLayer input) throws NullPointerException {
        return withOutputLayerUnchecked(Objects.requireNonNull(input));
    }

    public NetworkLayoutBuilder withInputLayer(int size) throws IllegalArgumentException{
        return withInputLayer(size, null, null);
    }

    public NetworkLayoutBuilder withInputLayer(int size, ActivationFunction activationFunction) throws IllegalArgumentException{
        return withInputLayer(size, null, activationFunction);
    }

    public NetworkLayoutBuilder withInputLayer(int size, InputNormalizer inputNormalizer) throws IllegalArgumentException{
        return withInputLayer(size, inputNormalizer, null);
    }

    public NetworkLayoutBuilder withInputLayer(int size, InputNormalizer inputNormalizer, ActivationFunction activationFunction) throws IllegalArgumentException{
        return withInputLayerUnchecked(new NetworkLayer(size, inputNormalizer, activationFunction));
    }


    private NetworkLayoutBuilder withOutputLayerUnchecked(NetworkLayer output) {
        this.output = output;
        layoutCached = false;
        return this;
    }

    public NetworkLayoutBuilder withOutputLayer(NetworkLayer output) throws NullPointerException {
        return withOutputLayerUnchecked(Objects.requireNonNull(output));
    }

    public NetworkLayoutBuilder withOutputLayer(int size) throws IllegalArgumentException {
        return withOutputLayer(size, null, null);
    }

    public NetworkLayoutBuilder withOutputLayer(int size, InputNormalizer inputNormalizer) throws IllegalArgumentException {
        return withOutputLayer(size, inputNormalizer, null);
    }

    public NetworkLayoutBuilder withOutputLayer(int size, ActivationFunction activationFunction) throws IllegalArgumentException {
        return withOutputLayer(size, null, activationFunction);
    }

    public NetworkLayoutBuilder withOutputLayer(int size, InputNormalizer inputNormalizer, ActivationFunction activationFunction) throws IllegalArgumentException {
        return withOutputLayerUnchecked(new NetworkLayer(size, inputNormalizer, activationFunction));
    }



    private NetworkLayoutBuilder addLayerUnchecked(NetworkLayer layer) {
        hiddenLayers.add(layer);
        if(layoutCached){
            layoutCached = false;
            layout = null;
        }
        return this;
    }

    public NetworkLayoutBuilder addLayer(NetworkLayer layer) throws NullPointerException {
        return addLayerUnchecked(Objects.requireNonNull(layer));
    }

    public NetworkLayoutBuilder addLayer(int size) throws IllegalArgumentException {
        return addLayer(size, null, null);
    }

    public NetworkLayoutBuilder addLayer(int size, ActivationFunction activationFunction) throws IllegalArgumentException {
        return addLayer(size, null, activationFunction);
    }

    public NetworkLayoutBuilder addLayer(int size, InputNormalizer inputNormalizer) throws IllegalArgumentException {
        return addLayer(size, inputNormalizer, null);
    }

    public NetworkLayoutBuilder addLayer(int size, InputNormalizer inputNormalizer, ActivationFunction activationFunction) throws IllegalArgumentException {
        return addLayerUnchecked(new NetworkLayer(size, inputNormalizer, activationFunction));
    }



    public NetworkLayoutBuilder addLayers(NetworkLayer layer1, NetworkLayer layer2, NetworkLayer... layers) throws NullPointerException {
        NetworkLayer[] allLayers = new NetworkLayer[2 + layers.length];
        int index = 0;
        allLayers[index] = Objects.requireNonNull(layer1);
        allLayers[++index] = Objects.requireNonNull(layer2);
        for(NetworkLayer layer : layers)
            allLayers[++index] = Objects.requireNonNull(layer);
        
        return addLayersUnchecked(allLayers);
    }

    public NetworkLayoutBuilder addLayers(NetworkLayer[] layers) throws NullPointerException {
        for(NetworkLayer layer : layers)
            Objects.requireNonNull(layer);

        return addLayersUnchecked(layers);
    }

    private NetworkLayoutBuilder addLayersUnchecked(NetworkLayer[] layers) throws NullPointerException {
        for(NetworkLayer layer : layers)
            addLayerUnchecked(layer);

        return this;
    }

    public NetworkLayoutBuilder addLayers(int numLayers, int size) throws IllegalArgumentException {
        return addLayers(numLayers, size, null, null);
    }

    public NetworkLayoutBuilder addLayers(int numLayers, int size, ActivationFunction activationFunction) throws IllegalArgumentException {
        return addLayers(numLayers, size, null, activationFunction);
    }

    public NetworkLayoutBuilder addLayers(int numLayers, int size, InputNormalizer inputNormalizer) throws IllegalArgumentException {
        return addLayers(numLayers, size, inputNormalizer, null);
    }

    public NetworkLayoutBuilder addLayers(int numLayers, int size, InputNormalizer inputNormalizer, ActivationFunction activationFunction) throws IllegalArgumentException {
        if(numLayers < 0) throw new IllegalArgumentException("Cannot add a negative amount of layers.");

        NetworkLayer[] layers = new NetworkLayer[numLayers];
        for(int i = 0; i < layers.length; i++)
            layers[i] = new NetworkLayer(size, inputNormalizer, activationFunction);

        return addLayersUnchecked(layers);
    }
}
package com.mjsd.simpleneuralnetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Function;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;

import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.ActivationFunction;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.InputNormalizer;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.InputProvider;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.OutputHandler;
import com.mjsd.simpleneuralnetwork.exceptions.DimensionsMismatchException;
import com.mjsd.simpleneuralnetwork.gson.CustomGsonFactory;

public class NeuralNetworkBuilder<E extends SimpleNeuralNetwork> {
    final private Function<NetworkLayout, E> NETWORK_SUPPLIER;
    private NetworkLayoutBuilder layoutBuilder;
    private List<InputProvider> inputProviders;
    private List<OutputHandler> outputHandlers;
    private double[][][] weights = null;
    private double[][] biases = null;
    private Optional<Boolean> parallelInputFetching = Optional.empty(),
                              parallelOutputHandling = Optional.empty(),
                              parallelForwardPass = Optional.empty();

    public NeuralNetworkBuilder(Function<NetworkLayout, E> supplier) throws NullPointerException{
        NETWORK_SUPPLIER = Objects.requireNonNull(supplier);
        layoutBuilder = new NetworkLayoutBuilder();
        inputProviders = null;
        outputHandlers = null;
    }

    public NeuralNetworkBuilder(Function<NetworkLayout, E> supplier, E initialState) throws NullPointerException{
        NETWORK_SUPPLIER = Objects.requireNonNull(supplier);
        Objects.requireNonNull(initialState);
        this.layoutBuilder = new NetworkLayoutBuilder(initialState.getLayout());
        inputProviders = initialState.getInputProviders();
        outputHandlers = initialState.getOutputHandlers();

        double[][][] initialWeights = initialState.getWeights();
        double[][] initialBiases = initialState.getBiases();
        this.weights = Arrays.copyOf(initialWeights, initialWeights.length);
        this.biases = Arrays.copyOf(initialBiases, initialBiases.length);
    }

    public NeuralNetworkBuilder(Function<NetworkLayout, E> supplier, NetworkLayout initialState) throws NullPointerException{
        this(supplier, initialState, null, null);
    }

    public NeuralNetworkBuilder(Function<NetworkLayout, E> supplier, NetworkLayout initialState, List<InputProvider> inputProvider, List<OutputHandler> outputHandler) throws NullPointerException{
        NETWORK_SUPPLIER = Objects.requireNonNull(supplier);
        this.layoutBuilder = new NetworkLayoutBuilder(initialState);
        withInputProviders(inputProvider);
        withOutputHandlers(outputHandler);
    }


    /**
     * Creates a NeuralNetwork based on the state of this builder.
     * @return A new NeuralNetwork object.
     * @throws IllegalStateException If {@code withInputLayer()} or {@code withOutputLayer()} are not called prior to {@code create()}.
     * @see #withInputLayer()
     * @see #withOutputLayer()
     */
    public E build() throws IllegalStateException {
        E instance = NETWORK_SUPPLIER.apply(layoutBuilder.build());
        if(parallelInputFetching.isPresent()) instance.setParallelInputFetching(parallelInputFetching.get());
        if(parallelOutputHandling.isPresent()) instance.setParallelOutputHandling(parallelOutputHandling.get());
        if(parallelForwardPass.isPresent()) instance.setParallelForwardPass(parallelForwardPass.get());

        try {
            if(inputProviders != null) instance.setInputProviders(inputProviders);
            if(outputHandlers != null) instance.setOutputHandlers(outputHandlers);

            boolean setBiases = biases != null,
                    setWeights = weights != null;

            if(setBiases) NeuralNetworkTools.ensureValidBiasArray(instance.LAYOUT, biases);
            if(setWeights) NeuralNetworkTools.ensureValidWeightArray(instance.LAYOUT, weights);

            if(setBiases) instance.biases = NeuralNetworkTools.deepCopy(biases);
            if(setWeights) instance.weights = NeuralNetworkTools.deepCopy(weights);

        } catch (IllegalArgumentException | DimensionsMismatchException e) {
            throw new IllegalStateException("Unable to build neural network from current state.", e);
        }

        return instance;
    }

    public void reset(){
        layoutBuilder.reset();
        inputProviders = null;
        outputHandlers = null;
        weights = null;
        biases = null;
    }

    public NeuralNetworkBuilder<E> withLayout(NetworkLayout layout) throws NullPointerException{
        layoutBuilder.setState(layout);
        return this;
    }

    public NeuralNetworkBuilder<E> withInputLayer(int size){
        layoutBuilder.withInputLayer(size);
        return this;
    }

    public NeuralNetworkBuilder<E> withInputLayer(int size, ActivationFunction activationFunction){
        layoutBuilder.withInputLayer(size, activationFunction);
        return this;
    }

    public NeuralNetworkBuilder<E> withInputLayer(int size, InputNormalizer inputNormalizer){
        layoutBuilder.withInputLayer(size, inputNormalizer);
        return this;
    }

    public NeuralNetworkBuilder<E> withInputLayer(int size, InputNormalizer inputNormalizer, ActivationFunction activationFunction){
        layoutBuilder.withInputLayer(size, inputNormalizer, activationFunction);
        return this;
    }

    public NeuralNetworkBuilder<E> withOutputLayer(int size){
        layoutBuilder.withOutputLayer(size);
        return this;
    }

    public NeuralNetworkBuilder<E> withOutputLayer(int size, InputNormalizer inputNormalizer){
        layoutBuilder.withOutputLayer(size, inputNormalizer);
        return this;
    }

    public NeuralNetworkBuilder<E> withOutputLayer(int size, ActivationFunction activationFunction){
        layoutBuilder.withOutputLayer(size, activationFunction);
        return this;
    }

    public NeuralNetworkBuilder<E> withOutputLayer(int size, InputNormalizer inputNormalizer, ActivationFunction activationFunction){
        layoutBuilder.withOutputLayer(size, inputNormalizer, activationFunction);
        return this;
    }

    public NeuralNetworkBuilder<E> addHiddenLayer(int size) throws IllegalArgumentException{
        layoutBuilder.addLayer(size);
        return this;
    }

    public NeuralNetworkBuilder<E> addHiddenLayer(int size, InputNormalizer inputNormalizer) throws IllegalArgumentException{
        layoutBuilder.addLayer(size, inputNormalizer);
        return this;
    }

    public NeuralNetworkBuilder<E> addHiddenLayer(int size, ActivationFunction activationFunction) throws IllegalArgumentException{
        layoutBuilder.addLayer(size, activationFunction);
        return this;
    }

    public NeuralNetworkBuilder<E> addHiddenLayer(int size, InputNormalizer inputNormalizer, ActivationFunction activationFunction) throws IllegalArgumentException{
        layoutBuilder.addLayer(size, inputNormalizer, activationFunction);
        return this;
    }

    public NeuralNetworkBuilder<E> addHiddenLayers(int numLayers, int size) throws IllegalArgumentException {
        layoutBuilder.addLayers(numLayers, size);
        return this;
    }

    public NeuralNetworkBuilder<E> addHiddenLayers(int numLayers, int size, InputNormalizer inputNormalizer) throws IllegalArgumentException{
        layoutBuilder.addLayers(numLayers, size, inputNormalizer);
        return this;
    }

    public NeuralNetworkBuilder<E> addHiddenLayers(int numLayers, int size, ActivationFunction activationFunction) throws IllegalArgumentException{
        layoutBuilder.addLayers(numLayers, size, activationFunction);
        return this;
    }

    public NeuralNetworkBuilder<E> addHiddenLayers(int numLayers, int size, InputNormalizer inputNormalizer, ActivationFunction activationFunction) throws IllegalArgumentException{
        layoutBuilder.addLayers(numLayers, size, inputNormalizer, activationFunction);
        return this;
    }

    public NeuralNetworkBuilder<E> withBiases(double[][] biases) {
        this.biases = biases;
        return this;
    }

    public NeuralNetworkBuilder<E> withWeights(double[][][] weights) {
        this.weights = weights;
        return this;
    }
    
    public NeuralNetworkBuilder<E> withInputProviders(List<InputProvider> inputProviders) {
        this.inputProviders = (inputProviders == null) ? null : new ArrayList<>(inputProviders);
        return this;
    }
    
    public NeuralNetworkBuilder<E> withoutInputProviders() {
        this.inputProviders = null;
        return this;
    }

    public NeuralNetworkBuilder<E> withOutputHandlers(List<OutputHandler> outputHandlers) {
        this.outputHandlers = (outputHandlers == null) ? null : new ArrayList<>(outputHandlers);
        return this;
    }

    public NeuralNetworkBuilder<E> withoutOutputHandlers() {
        this.outputHandlers = null;
        return this;
    }

    public NeuralNetworkBuilder<E> setParallelInputFetching(Boolean enabled){
        this.parallelInputFetching = Optional.ofNullable(enabled);
        return this;
    }

    public NeuralNetworkBuilder<E> setParallelOutputHandling(Boolean enabled){
        this.parallelOutputHandling = Optional.ofNullable(enabled);
        return this;
    }

    public NeuralNetworkBuilder<E> setParallelForwardPass(Boolean enabled){
        this.parallelForwardPass = Optional.ofNullable(enabled);
        return this;
    }

    public NetworkLayout getLayout() throws IllegalStateException {
        return layoutBuilder.build();
    }

    public static <T extends SimpleNeuralNetwork> T fromJson(String json, Class<T> classOfT) throws JsonSyntaxException{
        Gson gson = CustomGsonFactory.getInstance();
        return gson.fromJson(json, classOfT);
    }

}

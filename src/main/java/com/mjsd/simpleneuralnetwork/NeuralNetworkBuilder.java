package com.mjsd.simpleneuralnetwork;

import java.util.Arrays;
import java.util.Objects;
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
    private InputProvider inputProvider;
    private OutputHandler outputHandler;
    private double[][][] weights = null;
    private double[][] biases = null;

    public NeuralNetworkBuilder(Function<NetworkLayout, E> supplier) throws NullPointerException{
        NETWORK_SUPPLIER = Objects.requireNonNull(supplier);
        layoutBuilder = new NetworkLayoutBuilder();
        inputProvider = InputProvider.NO_PROVIDER;
        outputHandler = OutputHandler.NO_HANDLER;
    }

    public NeuralNetworkBuilder(Function<NetworkLayout, E> supplier, E initialState) throws NullPointerException{
        NETWORK_SUPPLIER = Objects.requireNonNull(supplier);
        Objects.requireNonNull(initialState);
        this.layoutBuilder = new NetworkLayoutBuilder(initialState.getLayout());
        inputProvider = initialState.getInputProvider();
        outputHandler = initialState.getOutputHandler();

        double[][][] initialWeights = initialState.getWeights();
        double[][] initialBiases = initialState.getBiases();
        this.weights = Arrays.copyOf(initialWeights, initialWeights.length);
        this.biases = Arrays.copyOf(initialBiases, initialBiases.length);
    }

    public NeuralNetworkBuilder(Function<NetworkLayout, E> supplier, NetworkLayout initialState) throws NullPointerException{
        this(supplier, initialState, null, null);
    }

    public NeuralNetworkBuilder(Function<NetworkLayout, E> supplier, NetworkLayout initialState, InputProvider inputProvider, OutputHandler outputHandler) throws NullPointerException{
        NETWORK_SUPPLIER = Objects.requireNonNull(supplier);
        this.layoutBuilder = new NetworkLayoutBuilder(initialState);
        this.inputProvider = InputProvider.ensureProvider(inputProvider);
        this.outputHandler = OutputHandler.ensureHandler(outputHandler);
    }


    /**
     * Creates a NeuralNetwork based on the state of this builder.
     * @return A new NeuralNetwork object.
     * @throws IllegalStateException If {@code withInputLayer()} or {@code withOutputLayer()} are not called prior to {@code create()}.
     * @see #withInputLayer()
     * @see #withOutputLayer()
     */
    public E build() throws IllegalStateException, DimensionsMismatchException {
        E instance = NETWORK_SUPPLIER.apply(layoutBuilder.build());
        if(biases != null) instance.biases = NeuralNetworkTools.deepCopy(NeuralNetworkTools.ensureValidBiasArray(instance.LAYOUT, biases));
        if(weights != null) instance.weights = NeuralNetworkTools.deepCopy(NeuralNetworkTools.ensureValidWeightArray(instance.LAYOUT, weights));
        instance.setInputProvider(inputProvider);
        instance.setOutputHandler(outputHandler);
        return instance;
    }

    public void reset(){
        layoutBuilder.reset();
        inputProvider = InputProvider.NO_PROVIDER;
        outputHandler = OutputHandler.NO_HANDLER;
        weights = null;
        biases = null;
    }

    public NeuralNetworkBuilder<E> withLayout(NetworkLayout layout) throws NullPointerException{
        this.layoutBuilder = new NetworkLayoutBuilder(layout);
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
    

    public void setInputProvider(InputProvider inputProvider) {
        this.inputProvider = (inputProvider == null) ? InputProvider.NO_PROVIDER : inputProvider;
    }

    public void setOutputHandler(OutputHandler outputHandler) {
        this.outputHandler = (outputHandler == null) ? OutputHandler.NO_HANDLER : outputHandler;
    }

    public NetworkLayout getLayout() throws IllegalStateException {
        return layoutBuilder.build();
    }

    public static <T extends SimpleNeuralNetwork> T fromJson(String json, Class<T> classOfT) throws JsonSyntaxException{
        Gson gson = CustomGsonFactory.getInstance();
        return gson.fromJson(json, classOfT);
    }

}

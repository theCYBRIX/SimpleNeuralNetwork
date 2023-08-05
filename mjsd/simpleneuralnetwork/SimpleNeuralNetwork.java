
package mjsd.simpleneuralnetwork;

import com.google.gson.Gson;
import com.google.gson.annotations.JsonAdapter;

import mjsd.simpleneuralnetwork.NetworkLayout.NetworkLayer;
import mjsd.simpleneuralnetwork.exceptions.DimensionsMismatchException;
import mjsd.simpleneuralnetwork.gson.*;
import mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;

import java.util.List;
import java.util.Objects;

@JsonAdapter(SimpleNeuralNetworkAdapter.class)
public class SimpleNeuralNetwork {

	final protected int OUTPUT_LAYER;
	final protected NetworkLayout LAYOUT;

	protected double[][][] weights;
	protected double[][] biases;

	protected double[][] hiddenLayers;
	protected ActivationFunction[] hiddenActivations;
	protected InputNormalizer[] hiddenNormalizers;

	protected double[] inputs;
	protected ActivationFunction inputActivation;
	protected InputNormalizer inputNormalizer;

	protected double[] outputs;
	protected ActivationFunction outputActivation;
	protected InputNormalizer outputNormalizer;

	private OutputHandler outputHandler;
	private InputProvider inputProvider;

	protected SimpleNeuralNetwork(NetworkLayout layout) throws NullPointerException{
		this(layout, null, null);
	}

	protected SimpleNeuralNetwork(SimpleNeuralNetwork template) throws NullPointerException{
		this(Objects.requireNonNull(template).LAYOUT, template.outputHandler, template.inputProvider);
		this.weights = NeuralNetworkTools.deepCopy(template.weights);
		this.biases = NeuralNetworkTools.deepCopy(template.biases);
	}

	
	protected SimpleNeuralNetwork(NetworkLayout layout, OutputHandler outputHandler, InputProvider inputProvider) throws NullPointerException {
		LAYOUT = Objects.requireNonNull(layout);

		NetworkLayer inputLayerLayout = layout.getInputLayer(),
					 outputLayerLayout = layout.getOutputLayer();
		List<NetworkLayer> hiddenLayerLayouts = layout.getHiddenLayers();

		this.outputHandler = OutputHandler.ensureHandler(outputHandler);
		this.inputProvider = InputProvider.ensureProvider(inputProvider);

		this.hiddenLayers = new double[hiddenLayerLayouts.size()][];
		this.hiddenActivations = new ActivationFunction[hiddenLayerLayouts.size()];
		this.hiddenNormalizers = new InputNormalizer[hiddenLayerLayouts.size()];
		for (int i = 0; i < hiddenLayerLayouts.size(); i++) {
			NetworkLayer layerLayout = hiddenLayerLayouts.get(i);
			this.hiddenLayers[i] = new double[layerLayout.getNodeCount()];
			this.hiddenActivations[i] = layerLayout.getActivationFunction();
			this.hiddenNormalizers[i] = layerLayout.getInputNormalizer();
		}

		this.inputs= new double[inputLayerLayout.getNodeCount()];
		this.inputActivation = inputLayerLayout.getActivationFunction();
		this.inputNormalizer = inputLayerLayout.getInputNormalizer();

		this.outputs = new double[outputLayerLayout.getNodeCount()];
		this.outputActivation = outputLayerLayout.getActivationFunction();
		this.outputNormalizer = outputLayerLayout.getInputNormalizer();

		this.weights = new double[hiddenLayers.length + 1][][];
		this.biases = new double[weights.length][];

		double[] previousLayer = inputs;
		for (int layer = 0; layer < hiddenLayers.length; layer++) {
			this.weights[layer] = new double[hiddenLayers[layer].length][];
			this.biases[layer] = new double[hiddenLayers[layer].length];
			for (int node = 0; node < weights[layer].length; node++)
				this.weights[layer][node] = new double[previousLayer.length];
			previousLayer = hiddenLayers[layer];
		}

		OUTPUT_LAYER = hiddenLayers.length;

		this.weights[OUTPUT_LAYER] = new double[outputs.length][];
		this.biases[OUTPUT_LAYER] = new double[outputs.length];
		for (int node = 0; node < outputs.length; node++)
			this.weights[OUTPUT_LAYER][node] = new double[previousLayer.length];

	}

	
	protected SimpleNeuralNetwork(NetworkLayout layout, double[][][] weights, double[][] biases, OutputHandler outputHandler, InputProvider inputProvider) throws NullPointerException {
		LAYOUT = Objects.requireNonNull(layout);
		this.weights = Objects.requireNonNull(weights);
		this.biases = Objects.requireNonNull(biases);

		NetworkLayer inputLayerLayout = layout.getInputLayer(),
					 outputLayerLayout = layout.getOutputLayer();
		List<NetworkLayer> hiddenLayerLayouts = layout.getHiddenLayers();

		this.outputHandler = OutputHandler.ensureHandler(outputHandler);
		this.inputProvider = InputProvider.ensureProvider(inputProvider);

		this.hiddenLayers = new double[hiddenLayerLayouts.size()][];
		this.hiddenActivations = new ActivationFunction[hiddenLayerLayouts.size()];
		this.hiddenNormalizers = new InputNormalizer[hiddenLayerLayouts.size()];
		for (int i = 0; i < hiddenLayerLayouts.size(); i++) {
			NetworkLayer layerLayout = hiddenLayerLayouts.get(i);
			this.hiddenLayers[i] = new double[layerLayout.getNodeCount()];
			this.hiddenActivations[i] = layerLayout.getActivationFunction();
			this.hiddenNormalizers[i] = layerLayout.getInputNormalizer();
		}

		this.inputs= new double[inputLayerLayout.getNodeCount()];
		this.inputActivation = inputLayerLayout.getActivationFunction();
		this.inputNormalizer = inputLayerLayout.getInputNormalizer();

		this.outputs = new double[outputLayerLayout.getNodeCount()];
		this.outputActivation = outputLayerLayout.getActivationFunction();
		this.outputNormalizer = outputLayerLayout.getInputNormalizer();

		OUTPUT_LAYER = hiddenLayers.length;
	}

	public void forwardPass(){
		inputProvider.getInputs(inputs);

		applyLayerModifiers(inputs, inputNormalizer, inputActivation);

		double[] previousLayer = inputs;

		for(int layer = 0; layer < hiddenLayers.length; layer++){
			hiddenLayers[layer] = vectorSum(dotSequence(previousLayer, weights[layer]), biases[layer]);
			applyLayerModifiers(hiddenLayers[layer], hiddenNormalizers[layer], hiddenActivations[layer]);
			previousLayer = hiddenLayers[layer];
		}

		outputs = vectorSum(dotSequence(previousLayer, weights[OUTPUT_LAYER]), biases[OUTPUT_LAYER]);
		applyLayerModifiers(outputs, outputNormalizer, outputActivation);
	
		outputHandler.handle(outputs);
	}

	private static double dotProduct(double[] v1, double[] v2) {
		double product = 0;

		for (int i = 0; i < v1.length; i++)
			product += v1[i] * v2[i];

		return product;
	}

	private static double[] dotSequence(double[] v1, double[][] crMatrix) {
		double product[] = new double[crMatrix.length];

		for (int column = 0; column < crMatrix.length; column++)
			product[column] = dotProduct(v1, crMatrix[column]);

		return product;
	}

	private static double[] vectorSum(double[] v1, double[] v2) {
		double sum[] = new double[v1.length];

		for (int i = 0; i < v1.length; i++)
			sum[i] = v1[i] + v2[i];

		return sum;
    }
	
	public SimpleNeuralNetwork copy() {
		return new SimpleNeuralNetwork(LAYOUT, NeuralNetworkTools.deepCopy(weights), NeuralNetworkTools.deepCopy(biases), outputHandler, inputProvider);
	}

	/**
	 * @return A NeuralNetworkBuilder initialized to the state of the this network.
	 */
	public NeuralNetworkBuilder<? extends SimpleNeuralNetwork> newBuilder(){
		return new SimpleNeuralNetworkBuilder(this);
	}

	@Override
	public String toString() {
		return SimpleNeuralNetwork.toString(this);
	}

	private static String toString(SimpleNeuralNetwork network){
		StringBuilder out = new StringBuilder();
		out.append("input: ");
		arrayToString(out, network.inputs);
		for(int i = 0, next = 1; i < network.hiddenLayers.length; next++){
			out.append("\nlayer ").append(next).append(": ");
			arrayToString(out, network.hiddenLayers[i]);
			i = next;
		}
		out.append("\noutput: ");
		arrayToString(out, network.outputs);
		return out.toString();
	}


    private static void arrayToString(StringBuilder appendTo, double[] array){
        appendTo.append("[").append(array[0]);
        for(int i = 1; i < array.length; i++)
            appendTo.append(", ").append(array[i]);
        appendTo.append("]");
    }

	@Override
	public boolean equals(Object obj) {
		if(!(obj instanceof SimpleNeuralNetwork)) return false;
		SimpleNeuralNetwork other = (SimpleNeuralNetwork)obj;

		if(!NeuralNetworkTools.haveSameLayout(this, other)) return false;

		double[][][] otherWeights = other.weights;

		for(int layer = 0; layer < weights.length; layer++)
			for(int node = 0; node < weights[layer].length; node++)
				for(int weight = 0; weight < weights[layer][node].length; weight++)
					if(this.weights[layer][node][weight] != otherWeights[layer][node][weight])
						return false;

		double[][] otherBiases = other.biases;

		for(int layer = 0; layer < biases.length; layer++)
			for(int node = 0; node < biases[layer].length; node++)
				if(this.biases[layer][node] != otherBiases[layer][node])
					return false;

		return true;
	}
	






	/*******************************************************************************************************************
	***************************************************** Getters ******************************************************
	*******************************************************************************************************************/

	public double getInput(int index) throws IndexOutOfBoundsException {
		return inputs[index];
	}

	public double[] getInputLayer() {
		return inputs;
	}

	public InputProvider getInputProvider() {
		return inputProvider;
	}

	public double getOutput(int index) throws IndexOutOfBoundsException {
		return outputs[index];
	}

	public double[] getOutputLayer() {
		return outputs;
	}

	public OutputHandler getOutputHandler() {
		return outputHandler;
	}

	public NetworkLayout getLayout(){
		return LAYOUT;
	}

	protected NetworkLayer getHiddenLayerLayout(int layerIndex) throws IndexOutOfBoundsException {
		return LAYOUT.HIDDEN_LAYERS.get(layerIndex);
	}

	public double getValue(int hiddenLayerIndex, int nodeIndex) throws ArrayIndexOutOfBoundsException {
		return hiddenLayers[hiddenLayerIndex][nodeIndex];
	}

    public String toJson(){
        return this.toJson(CustomGsonFactory.getInstance());
    }

    public String toJson(Gson gson){
        return gson.toJson(this, SimpleNeuralNetwork.class);
    }

	/**
	 * Gets the 2D array containing the values of the Neural Network's
	 * nodes in the form of {@code hiddenLayers[layer][node]}.
	 * 
	 * @return An array that functions as a column row matrix.
	 * @apiNote The returned array is not a copy. The neural network will modify it's contents with each call to {@link #forwardPass()}.
	 */
	protected double[][] getHiddenLayers() {
		return hiddenLayers;
	}






	/*******************************************************************************************************************
	***************************************************** Setters ******************************************************
	*******************************************************************************************************************/

	public void setInput(double[] values) throws NullPointerException, DimensionsMismatchException {
		if (Objects.requireNonNull(values).length != this.inputs.length)
			throw new DimensionsMismatchException("Number of values (" + values.length + ") does not match number of input nodes ("
					+ this.inputs.length + ").");

		synchronized(this.inputs){
			for (int i = 0; i < this.inputs.length; i++)
				this.inputs[i] = values[i];
		}
	}

	public void setInput(int index, double value) {
		inputs[index] = value;
	}

	public void setOutputHandler(OutputHandler outputHandler) {
		synchronized (this.outputHandler) {
			this.outputHandler = OutputHandler.ensureHandler(outputHandler);
		}
	}

	public void setInputProvider(InputProvider inputProvider) {
		synchronized (this.inputProvider) {
			this.inputProvider = InputProvider.ensureProvider(inputProvider);
		}
	}



	/**
	 * @return a deep copy of the 3-dimensional array containing the weights of the network; consisting of {@code weights[layer][node][weight]}.
	 * @see {@link MutableNeuralNetwork#retrieveWeightsArray() }
	 */
	final public double[][][] getWeights()  {
		return NeuralNetworkTools.deepCopy(weights);
	}
	
	/**
	 * @return A deep copy of the 2-dimensional array containing the biases of this network; consisting of {@code biases[layer][node]}.
	 * @see {@link MutableNeuralNetwork#retrieveBiasesArray() }
	 */
	final public double[][] getBiases(){
		return NeuralNetworkTools.deepCopy(biases);
	}





	/*******************************************************************************************************************
	************************************************* Static Methods ***************************************************
	*******************************************************************************************************************/

	private static void applyLayerModifiers(double[] layer, InputNormalizer normalizer, ActivationFunction activationFunction){
		normalizer.normalize(layer);

		for(int i = 0; i < layer.length; i++)
			layer[i] = activationFunction.apply(layer, i);
	}

	@JsonAdapter(ActivationFunctionAdapter.class)
	@FunctionalInterface
	public interface ActivationFunction{
		public double apply(double[] layer, int index);

        public static ActivationFunction ensureFunction(ActivationFunction function){
            if(function == null)
                return ActivationFunctions.LINEAR;
            else
                return function;
        }
	}

	@JsonAdapter(InputNormalizerAdapter.class)
	@FunctionalInterface
	public interface InputNormalizer {
		public void normalize(double[] values);

        public static InputNormalizer ensureNormalizer(InputNormalizer normalizer){
            if(normalizer == null)
                return InputNormalizers.NO_NORMALIZER;
            else
                return normalizer;
        }
	}

	@JsonAdapter(OutputHandlerAdapter.class)
	@FunctionalInterface
	public interface OutputHandler {
		public static String NO_HANDLER_STRING = "NO_HANDLER";

		public void handle(double[] outputs);

		public static OutputHandler ensureHandler(OutputHandler handler){
			if(handler == null)
				return OutputHandler.NO_HANDLER;
			else
				return handler;
		}

		final public static OutputHandler NO_HANDLER = new OutputHandler() {
			@Override public void handle(double[] values){}
			@Override public String toString(){ return NO_HANDLER_STRING; }
		};
	}

	@JsonAdapter(InputProviderAdapter.class)
	@FunctionalInterface
	public interface InputProvider {
		public static String NO_PROVIDER_STRING = "NO_PROVIDER";

		public void getInputs(double[] inputs);

		public static InputProvider ensureProvider(InputProvider provider){
			if(provider == null)
				return InputProvider.NO_PROVIDER;
			else
				return provider;
		}

		final public static InputProvider NO_PROVIDER = new InputProvider() {
			@Override public void getInputs(double[] inputs){}
			@Override public String toString(){ return NO_PROVIDER_STRING; }
		};
	}
}

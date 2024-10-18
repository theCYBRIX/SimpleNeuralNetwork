package com.github.thecybrix.simpleneuralnetwork.api;

import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.logging.ConsoleHandler;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;
import java.util.stream.Collectors;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayout;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkBuilder;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkTools;
import com.github.thecybrix.simpleneuralnetwork.exceptions.DimensionsMismatchException;
import com.github.thecybrix.simpleneuralnetwork.serialization.json.CustomGsonFactory;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.EvolutionaryTrainer;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.NetworkEvolutionManager;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelector;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ValueMappingTrainer;
import com.github.thecybrix.simpleneuralnetwork.training.simple.SimpleEvolutionManager;
import com.github.thecybrix.util.CallbackInvoker;
import com.github.thecybrix.util.Fraction;
import com.github.thecybrix.util.LELengthPrefixedReader;
import com.github.thecybrix.util.LELengthPrefixedWriter;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public class APIIOHandler<E extends MutableNeuralNetwork> implements CallbackInvoker<Exception>{
    final private static Logger LOGGER = Logger.getLogger(APIIOHandler.class.getName());

    static {
        class PrintlnFormatter extends Formatter{
            @Override
            public String format(LogRecord record) {
                return record.getMessage() + "\n";
            }
        }

        try {
            Logger rootLogger = Logger.getLogger("");
            rootLogger.removeHandler(rootLogger.getHandlers()[0]);

            FileHandler logFileHandler = new FileHandler("TestSaves\\APIIOHandler.log", false);
            logFileHandler.setFormatter(new SimpleFormatter());
            logFileHandler.setLevel(Level.ALL);
            LOGGER.addHandler(logFileHandler);

            ConsoleHandler consoleHandler = new ConsoleHandler();
            consoleHandler.setFormatter(new PrintlnFormatter());
            consoleHandler.setLevel(Level.INFO);
            LOGGER.addHandler(consoleHandler);

            LOGGER.setLevel(Level.ALL);
        } catch (Exception e) {
           LOGGER.severe("Failed to initialize log handler.");
        }
        
    }

    public enum ParentSelection {
        ROULETTE_WHEEL_PREFER_LARGE,
        ROULETTE_WHEEL_PREFER_SMALL,
        TOURNAMENT_PREFER_LARGE,
        TOURNAMENT_PREFER_SMALL,
        ELITES_PREFER_LARGE,
        ELITES_PREFER_SMALL;
    }

    public enum State {
        RUNNING,
        SUCCESS,
        CANCELLED,
        FAILED
    }

    public enum Status {
        OK,
        ERROR;
    }

    final private LinkedList<Consumer<Exception>> CALLBACKS = new LinkedList<>();
    final private LinkedList<RequestHandler<E>> CUSTOM_REQUEST_HANDLERS = new LinkedList<>();

    final private Gson GSON;

    final private ExecutorService EXECUTOR_SERVICE;
    final private NeuralNetworkBuilder<E> NETWORK_BUILDER;

    final private Object PREV_GEN_LOCK = new Object();
    final private Object CURRENT_GEN_LOCK = new Object();

    private List<ScoredNetwork<E>> previousGeneration = Collections.emptyList();

    private NetworkEvolutionManager<E> evolutionManager;
    private ParentSelector<E> parentSelector;

    private Future<?> dataTrainedNetworks;
    private EvolutionaryTrainer<E> datasetTrainer;
    private int numTrainingSamples;
    private Instant trainingStartTime, trainingEndTime;
    
    private volatile boolean keepAlive = false;
    private int numNetworks;
    private List<ScoredNetwork<E>> neuralNetworks = new ArrayList<>();


    public APIIOHandler(NeuralNetworkBuilder<E> networkBuilder) throws IllegalArgumentException, NullPointerException {
        this(networkBuilder, null, null);
    }

    public APIIOHandler(NeuralNetworkBuilder<E> networkBuilder, ExecutorService executorService) throws IllegalArgumentException, NullPointerException {
        this(networkBuilder, null, executorService);
    }

    public APIIOHandler(NeuralNetworkBuilder<E> networkBuilder, ParentSelector<E> parentSelector) throws IllegalArgumentException, NullPointerException {
        this(networkBuilder, parentSelector, null);
    }

    public APIIOHandler(NeuralNetworkBuilder<E> networkBuilder, ParentSelector<E> parentSelector, ExecutorService executorService) throws IllegalArgumentException, NullPointerException {
        NETWORK_BUILDER = Objects.requireNonNull(networkBuilder, "Network builder is null.");
        EXECUTOR_SERVICE = (executorService != null) ? executorService : Executors.newWorkStealingPool();
        this.parentSelector = (parentSelector != null) ? parentSelector : ParentSelector.eliteSelection();

        GsonBuilder gsonBuilder = CustomGsonFactory.getInstance().newBuilder();
        gsonBuilder.enableComplexMapKeySerialization();
        gsonBuilder.serializeSpecialFloatingPointValues();
        gsonBuilder.setLenient();
        GSON = gsonBuilder.create();
    }


    public void handle(InputStream input, OutputStream output) throws InterruptedException {
        try(
            LELengthPrefixedReader reader = new LELengthPrefixedReader(input);
            LELengthPrefixedWriter writer = new LELengthPrefixedWriter(output);
        ) {
            keepAlive = true;
            while(keepAlive){
                if(Thread.interrupted()) throw new InterruptedException();

                String request = reader.readString();
                LOGGER.finest(() -> "Request received:\n" + request);

                String response = GSON.toJson(handleRequest(request));
                LOGGER.finest(() -> "Response packet:\n" + response);

                writer.writeString(response);
                writer.flush();
            }
        } catch (Exception e) {
            logError(e);
        }
    }

    public void stop(){
        keepAlive = false;
    }

    private void logError(Exception e){
        LOGGER.warning(e.getMessage());
        LOGGER.fine(stackTraceToString(e));
        processCallbacks(e);
    }

    private ResponsePacket<E> handleRequest(String request){
        try {
            RequestPacket r = GSON.fromJson(request, RequestPacket.class);

            switch (r.getRequest()) {
                case "setup":
                    setup(r.getSetupProperties());
                    return ResponsePacket.ok();

                case "randomize_networks":
                    synchronized(CURRENT_GEN_LOCK){
                        for (ScoredNetwork<E> scoredNetwork : neuralNetworks)
                            NeuralNetworkTools.randomizeWeightsAndBiases(scoredNetwork.get());
                        if(neuralNetworks.size() < numNetworks)
                            neuralNetworks.addAll(evolutionManager.createRandomGeneration(numNetworks - neuralNetworks.size()));
                    }
                    return ResponsePacket.ok();

                case "process_inputs":
                    NetworkDataPacket outputs = processInputs(r.getInputs());
                    return ResponsePacket.message(outputs);

                case "approximate_data_set":
                    if(datasetTrainer != null && datasetTrainer.isRunning()){
                        datasetTrainer.stop();
                        dataTrainedNetworks.get();
                    }
                    approximateDataSet(r.getTrainingData());
                    return ResponsePacket.ok();

                case "get_training_state":
                        if(dataTrainedNetworks == null) throw new IllegalStateException("Trainer has not been initialized.");

                        TrainingStatusPacket trainingStatus;
                        State state;
                        if(dataTrainedNetworks.isDone())
                            state = dataTrainedNetworks.isCancelled() ? State.CANCELLED : State.SUCCESS;
                        else
                            state = State.RUNNING;
                        long elapsedTime = trainingStartTime.until((trainingEndTime != null) ? trainingEndTime : Instant.now(), ChronoUnit.MILLIS);
                        double bestScore = datasetTrainer.getPreviousGeneration()
                                            .parallelStream()
                                            .filter(x -> x.getScore().isPresent())
                                            .mapToDouble(x -> x.getScore().getAsDouble())
                                            .min()
                                            .orElse(Double.NaN);

                        trainingStatus = new TrainingStatusPacket(state.toString(), elapsedTime, bestScore / numTrainingSamples, datasetTrainer.getGeneration());

                    return ResponsePacket.message(trainingStatus);

                case "stop_training":
                    if(datasetTrainer == null) throw new IllegalStateException("Trainer has not been initialized.");
                    if(datasetTrainer.isRunning()){
                        datasetTrainer.stop();
                        dataTrainedNetworks.get();
                    }
                        
                    return ResponsePacket.ok();
            
                case "create_new_generation":
                    createNewGeneration(r.getScores());
                    return ResponsePacket.ok();

                case "get_best_networks":
                    if(previousGeneration.isEmpty())
                        throw new IllegalStateException("No network heirarchy available. This can happen if the server has not created a new generation since it was initialized or reconfigured.");
                    return ResponsePacket.message( (r.numRequested > 0) ? getBestNetworks(r.numRequested) : getBestNetworks() );

                default:
                    for (RequestHandler<E> handler : CUSTOM_REQUEST_HANDLERS)
                        if(handler.isApplicable(r))
                            return handler.handle(r);

                    return ResponsePacket.error("Invalid request.", "\"" + r.getRequest() + "\" is not a recognized command.");
            }
        } catch (Exception e) {
            LOGGER.warning("Failed to handle request: " + e.getClass().getSimpleName());
            logError(e);
            return ResponsePacket.error(e.getClass().getSimpleName(), e.getMessage());
        }


    }
    
    public void addRequestHandler(RequestHandler<E> handler){
        CUSTOM_REQUEST_HANDLERS.add(handler);
    }
    
    public boolean removeRequestHandler(RequestHandler<E> handler){
        return CUSTOM_REQUEST_HANDLERS.removeLastOccurrence(handler);
    }

    public List<E> getBestNetworks(){
        synchronized(PREV_GEN_LOCK){
            return unpackSuppliers(previousGeneration);
        }
    }

    public List<E> getBestNetworks(int numNetworks) throws IllegalArgumentException, IndexOutOfBoundsException {
        synchronized(PREV_GEN_LOCK){
            int genSize = previousGeneration.size();
            return unpackSuppliers(previousGeneration.subList(genSize - numNetworks, genSize));
        }
    }


    public List<E> getPreviousGeneration(int from, int to) throws ArrayIndexOutOfBoundsException {
        synchronized(PREV_GEN_LOCK){
            return unpackSuppliers(previousGeneration.subList(from, to));
        }
    }

    private static <E extends MutableNeuralNetwork> List<E> unpackSuppliers(List<? extends Supplier<E>> list){
        return list.parallelStream()
                    .map(x -> x.get())
                    .collect(Collectors.toList());
    }

    private void setParentSelection(ParentSelection selector){
        switch (selector) {
            case ELITES_PREFER_LARGE:
                parentSelector = ParentSelector.eliteSelection();
                break;
            case ELITES_PREFER_SMALL:
                parentSelector = ParentSelector.eliteSelection(Comparator.reverseOrder());
                break;
            case ROULETTE_WHEEL_PREFER_LARGE:
                parentSelector = ParentSelector.rouletteWheelSelection();
                break;
            case ROULETTE_WHEEL_PREFER_SMALL:
                parentSelector = ParentSelector.rouletteWheelSelection(true);
                break;
            case TOURNAMENT_PREFER_LARGE:
                parentSelector = ParentSelector.tournamentSelection(Fraction.of(10, 100));
                break;
            case TOURNAMENT_PREFER_SMALL:
                parentSelector = ParentSelector.tournamentSelection(Fraction.of(10, 100), Comparator.reverseOrder());
                break;
            default:
                break;
        }
    }

    //TODO: make redundant or inferable information optional
    private void setup(SetupPacket properties) throws NoSuchElementException, DimensionsMismatchException, NullPointerException {
        NetworkLayout layout = properties.getLayout();
        List<MutableNeuralNetwork> initialNetworks = null;

        try {
            initialNetworks = properties.getNetworks();
            NeuralNetworkTools.requireSameDimensions(initialNetworks);
        } catch (NoSuchElementException e) {}

        try {
            setParentSelection(properties.getParentSelector());
        } catch (NoSuchElementException e) {}
        
        numNetworks = properties.getNumNetworks();
        NETWORK_BUILDER.reset().withLayout(layout);
        evolutionManager = new SimpleEvolutionManager<>(NETWORK_BUILDER::build, parentSelector);

        synchronized(PREV_GEN_LOCK){
            if(previousGeneration.size() > 0)
                previousGeneration = Collections.emptyList();
        }

        synchronized(CURRENT_GEN_LOCK){
            neuralNetworks = new ArrayList<>();
            
            if(initialNetworks != null)
                neuralNetworks.addAll(
                    initialNetworks.parallelStream()
                    .map(x -> new ScoredNetwork<E>(NETWORK_BUILDER.convert(x)))
                    .collect(Collectors.toList())
                );
                
            neuralNetworks.addAll(evolutionManager.createRandomGeneration(numNetworks - neuralNetworks.size()));
        }
        
    }

    private void approximateDataSet(TrainingDataSet dataSet) throws IllegalArgumentException, NullPointerException{
        ValueMappingTrainer<E> trainingScenario = ValueMappingTrainer.of(dataSet.inputs, dataSet.outputs, new ValueMappingTrainer.MeanSquaredError(), EXECUTOR_SERVICE);
        datasetTrainer = new EvolutionaryTrainer<>(numNetworks, evolutionManager, trainingScenario);
        datasetTrainer.attachCallback(trainer -> {
            setPrevGen(trainer.getPreviousGeneration());
            setCurrentGen(trainer.getNetworks());
        });
        datasetTrainer.addAllScored(neuralNetworks);
        numTrainingSamples = dataSet.inputs.length;
        trainingEndTime = null;
        dataTrainedNetworks = EXECUTOR_SERVICE.submit(() -> {
            datasetTrainer.run();
            trainingEndTime = Instant.now();
            setPrevGen(datasetTrainer.getPreviousGeneration());
            setCurrentGen(new ArrayList<>(datasetTrainer.getNetworks()));
        });
        trainingStartTime = Instant.now();
    }

    private NetworkDataPacket processInputs(NetworkDataPacket inputData){
        ArrayList<Callable<Void>> tasks = new ArrayList<>(inputData.size());
        NetworkDataPacket outputs = new NetworkDataPacket(inputData.size());
        
        synchronized(CURRENT_GEN_LOCK){
            for (Entry<Integer, double[]> item : inputData.entrySet()) {
                tasks.add(() -> { process(item.getKey(), item.getValue(), outputs); return null; });
            }

            try {
                EXECUTOR_SERVICE.invokeAll(tasks);
            } catch (InterruptedException e) {
            } catch (NullPointerException | RejectedExecutionException e) {
                e.printStackTrace();
            }
        }

        return outputs;
    }

    private void process(int networkIndex, double[] inputs, NetworkDataPacket results) throws DimensionsMismatchException, NullPointerException{
        E network = neuralNetworks.get(networkIndex).get();
        synchronized(network){
            network.setInputs(inputs);
            network.forwardPass();
            synchronized(results){
                results.put(networkIndex, network.getOutputs());
            }
        }
    }

    private void createNewGeneration(NetworkScorePacket scores){
        synchronized(CURRENT_GEN_LOCK){
            for (Entry<Integer, Double> packet : scores.entrySet())
                neuralNetworks.get(packet.getKey()).setScore(packet.getValue());

            ArrayList<ScoredNetwork<E>> newGeneration = new ArrayList<>(evolutionManager.createNewGeneration(neuralNetworks, numNetworks));

            synchronized(PREV_GEN_LOCK){
                previousGeneration = neuralNetworks;
                neuralNetworks = newGeneration;
            }
        }
    }

    private void setPrevGen(List<ScoredNetwork<E>> networks){
        synchronized(PREV_GEN_LOCK){
            previousGeneration = networks;
        }
    }

    private void setCurrentGen(List<ScoredNetwork<E>> networks){
        synchronized(CURRENT_GEN_LOCK){
            neuralNetworks = networks;
        }
    }

    public List<ScoredNetwork<E>> getPreviousGeneration(){
        return Collections.unmodifiableList(previousGeneration);
    }

    private static <T> T requireNonNull(T obj, String name) throws NoSuchElementException {
        if(obj == null) throw new NoSuchElementException("Field \"" + name + "\" not found.");
        return obj;
    }

    private static String stackTraceToString(Exception e){
        StringWriter stackTrace = new StringWriter();
        e.printStackTrace(new PrintWriter(stackTrace));
        return stackTrace.toString();
    }

    public static interface RequestHandler<E extends MutableNeuralNetwork>{
        public boolean isApplicable(RequestPacket request);
        public ResponsePacket<E> handle(RequestPacket request) throws Exception;
    }

    @SuppressWarnings("unused")
    public static class ResponsePacket<E extends MutableNeuralNetwork>{
        private Status status;
        private String message;
        private String details;
        private Map<Integer, double[]> networkOutputs;
        private List<E> networks;
        private TrainingStatusPacket trainingStatus;

        private ResponsePacket() {
            this(Status.OK);
        }

        private ResponsePacket(Status status) {
            this(status, null);
        }

        private ResponsePacket(Status status, String message) {
            this(status, message, null);
        }

        private ResponsePacket(Status status, String message, String details) {
            this(status, message, details, null, null, null);
        }

        private ResponsePacket(NetworkDataPacket networkOutputs) {
            this(Status.OK, null, null, networkOutputs, null, null);
        }

        private ResponsePacket(TrainingStatusPacket trainingStatus) {
            this(Status.OK, null, null, null, trainingStatus, null);
        }

        private ResponsePacket(List<E> networks) {
            this(Status.OK, null, null, null, null, networks);
        }

        private ResponsePacket(Status status, String message, String details, NetworkDataPacket networkOutputs, TrainingStatusPacket trainingStatus, List<E> networks) {
            this.status = status;
            this.message = message;
            this.details = details;
            this.networkOutputs = networkOutputs;
            this.networks = networks;
            this.trainingStatus = trainingStatus;
        }

        public static <E extends MutableNeuralNetwork> ResponsePacket<E> message(NetworkDataPacket data){
            return new ResponsePacket<>(data);
        }

        public static <E extends MutableNeuralNetwork> ResponsePacket<E> message(TrainingStatusPacket status){
            return new ResponsePacket<>(status);
        }

        public static <E extends MutableNeuralNetwork> ResponsePacket<E> message(List<E> data){
            return new ResponsePacket<>(data);
        }

        public static <E extends MutableNeuralNetwork> ResponsePacket<E> message(String message){
            return new ResponsePacket<>(Status.OK, message);
        }

        public static <E extends MutableNeuralNetwork> ResponsePacket<E> ok(){
            return new ResponsePacket<>();
        }
        
        public static <E extends MutableNeuralNetwork> ResponsePacket<E> error(Exception e){
            return error(e.getMessage(), e);
        }
        
        public static <E extends MutableNeuralNetwork> ResponsePacket<E> error(String message){
            return new ResponsePacket<>(Status.ERROR, message);
        }
        
        public static <E extends MutableNeuralNetwork> ResponsePacket<E> error(String message, Exception e){
            return error(message, stackTraceToString(e));
        }
        
        public static <E extends MutableNeuralNetwork> ResponsePacket<E> error(String message, String details){
            return new ResponsePacket<>(Status.ERROR, message, details);
        }
    }

    public static class RequestPacket{
        private String request;
        private NetworkDataPacket networkInputs;
        private NetworkScorePacket networkScores;
        private TrainingDataSet trainingDataSet;
        private SetupPacket setupProperties;
        private int numRequested = -1;
        
        public String getRequest() throws NoSuchElementException {
            return requireNonNull(request, "request");
        }
        
        public NetworkDataPacket getInputs() throws NoSuchElementException {
            return requireNonNull(networkInputs, "networkInputs");
        }

        public NetworkScorePacket getScores() throws NoSuchElementException {
            return requireNonNull(networkScores, "networkScores");
        }

        public SetupPacket getSetupProperties() throws NoSuchElementException {
            return requireNonNull(setupProperties, "setupProperties");
        }

        public TrainingDataSet getTrainingData() throws NoSuchElementException {
            return requireNonNull(trainingDataSet, "trainingDataSet");
        }
    }

    private static class SetupPacket{
        int numNetworks;
        ParentSelection parentSelector;
        NetworkLayout layout;
        List<MutableNeuralNetwork> initialNetworks;

        public int getNumNetworks() throws NoSuchElementException {
            return requireNonNull(numNetworks, "numNetworks");
        }

        public ParentSelection getParentSelector() throws NoSuchElementException {
            return requireNonNull(parentSelector, "comparator");
        }

        public NetworkLayout getLayout() throws NoSuchElementException {
            return requireNonNull(layout, "layout");
        }

        public List<MutableNeuralNetwork> getNetworks() throws NoSuchElementException {
            return requireNonNull(initialNetworks, "initialNetworks");
        }
    }
    
    @SuppressWarnings("unused")
    private static class NetworkScorePacket extends HashMap<Integer, Double>{
        public NetworkScorePacket(){
            super();
        }

        public NetworkScorePacket(int initialSize){
            super(initialSize);
        }
    }
    
    @SuppressWarnings("unused")
    private static class NetworkDataPacket extends HashMap<Integer, double[]>{
        public NetworkDataPacket(){
            super();
        }

        public NetworkDataPacket(int initialSize){
            super(initialSize);
        }
    }
    
    @SuppressWarnings("unused")
    private static class TrainingDataSet {
        double[][] inputs;
        double[][] outputs;
    }
    
    @SuppressWarnings("unused")
    private static class TrainingStatusPacket {
        private String state;
        private long elapsedTimeMS;
        private double averageError;
        private int generation;

        public TrainingStatusPacket(String state, long elapsedTimeMS, double averageError, int generation) {
            this.state = state;
            this.elapsedTimeMS = elapsedTimeMS;
            this.averageError = averageError;
            this.generation = generation;
        }
    }

    @Override
    public List<Consumer<Exception>> getCallbackList() {
        return CALLBACKS;
    }
    
}

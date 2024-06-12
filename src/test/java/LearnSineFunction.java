
import java.io.Console;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.DecimalFormat;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Scanner;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;

import com.google.gson.JsonParseException;
import com.mjsd.simpleneuralnetwork.ActivationFunctions;
import com.mjsd.simpleneuralnetwork.LossFunctions;
import com.mjsd.simpleneuralnetwork.NeuralNetworkBuilder;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetworkBuilder;
import com.mjsd.simpleneuralnetwork.training.evolution.TrainingScenario;
import com.mjsd.simpleneuralnetwork.training.RankedNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.RankedNeuralNetworkBuilder;
import com.mjsd.simpleneuralnetwork.training.evolution.SimpleEvolutionaryTrainer;

final public class LearnSineFunction extends TestingEnvironment {
    
    //Settings
    private static float fpsLimit = 0.0f;
    private static String savePath = "SinNetwork";
    private static FileType saveType = FileType.JSON;
    private static int networksPerGeneration = 10;

    private static int numSamples = 128;
    private static float acceptableError = 0.000005f;
    private static double[] limits = new double[]{0, 2 * Math.PI};

    private static NeuralNetworkBuilder<RankedNeuralNetwork> layout = new RankedNeuralNetworkBuilder()
                                                                      .withInputLayer(1)
                                                                      .withOutputLayer(1)
                                                                      .addHiddenLayers(2, 4, ActivationFunctions.TANH);

    //Class
    final private Console CONSOLE;
    private XYChart chart;
    private SwingWrapper<XYChart> chartWrapper;

    private static long frameTime = fpsLimit <= 0 ? 0l : Math.round(1000 / fpsLimit);

    final private Object SYNCH_OBJECT = new Object();
    final private ScheduledExecutorService EXECUTOR_SERVICE = Executors.newScheduledThreadPool(3);
    final private DecimalFormat DECIMAL_FORMAT = new DecimalFormat("0.00#####");
    final private DateTimeFormatter DATE_TIME_FORMAT = DateTimeFormatter.ofPattern("h':'mm a");
    
    private Thread mainThread, trainingThread;
    private SimpleEvolutionaryTrainer<RankedNeuralNetwork> trainer;

    private Instant startTime, endTime;
    private Optional<Double> bestScore = Optional.empty();

    final double[] x, y, error;

    public static void main(String[] args) {
        LearnSineFunction learnSineFunction = new LearnSineFunction(System.console());
        learnSineFunction.run();
    }

    public void run(){
        mainThread = Thread.currentThread();
        println("Network Layout:\n" + layout.getLayout());

        try{ printNetworkPredictions(savePath, saveType, x, y); } catch (Exception e){ e.printStackTrace(); }

        chart = new XYChartBuilder()
                    .title("Learn Sine Function")
                    .width(800)
                    .height(600)
                    .xAxisTitle("x")
                    .yAxisTitle("y")
                    .build();

        chart.addSeries("Sine", x, y, null);
        chart.addSeries("Prediction", x, new double[x.length], null);
        chartWrapper = new SwingWrapper<>(chart);
        chartWrapper.displayChart();

        trainer = new SimpleEvolutionaryTrainer<>(networksPerGeneration, layout::build, (a, b) -> 0 - a.compareTo(b), new ValueMappingTrainer(x, y));
        trainer.getPopulation().setParallel(true);

        trainingThread = new Thread(trainer);
        trainingThread.start();

        startTime = Instant.now();

        ScheduledFuture<?> graphUpdates = EXECUTOR_SERVICE.scheduleAtFixedRate(new GraphUpdater(), 50, 100, TimeUnit.MILLISECONDS);
        ScheduledFuture<?> statusUpdates = EXECUTOR_SERVICE.scheduleAtFixedRate(new StatusUpdater(), 50, 500, TimeUnit.MILLISECONDS);
        EXECUTOR_SERVICE.scheduleAtFixedRate(new InputWatcher(), 20, 100, TimeUnit.MILLISECONDS);

        try {
            synchronized(SYNCH_OBJECT){
                SYNCH_OBJECT.wait();
            }
        } catch (Exception exception) {
            if(!(exception instanceof InterruptedException)) exception.printStackTrace();
            println("Closing application... (Runtime: " + formatTime(Duration.between(startTime, Instant.now())) + " )");
            trainer.stop();
            return;

        } finally {
            graphUpdates.cancel(false);
            statusUpdates.cancel(false);
        }

        endTime = Instant.now();

        StringBuilder results = new StringBuilder();

        Optional<Double> bestScore = trainer.getPopulation().getBestScore();
        SimpleNeuralNetwork bestNetwork = getBestNetwork();

        results.append((bestScore.get() <= acceptableError) ? "Network reached desired proficiency in " : "Training stopped after ")
               .append(formatTime(Duration.between(startTime, endTime)))
               .append(".\nAverage error: ")
               .append((bestScore.isPresent() ? DECIMAL_FORMAT.format(bestScore.get()) : "< " + acceptableError))
               .append("\nNetwork Generation: ")
               .append(trainer.getGeneration())
               .append("\n\n");

        for(int i = 0; i < x.length; i++){
            bestNetwork.setInput(0, x[i]);
            bestNetwork.forwardPass();
            results.append(", ")
                   .append(DECIMAL_FORMAT.format(x[i]))
                   .append(", ")
                   .append(DECIMAL_FORMAT.format(y[i]))
                   .append(", ")
                   .append(DECIMAL_FORMAT.format(bestNetwork.getOutput(0)));
        }

        println(results.toString());
    }

    private LearnSineFunction(Console console){
        CONSOLE = Objects.requireNonNull(console, "Console is null.");

        double range = limits[1] - limits[0];
        x = new double[numSamples]; 
        y = new double[numSamples];
        error = new double[numSamples];

        double stepSize = range / (double)numSamples;
        for(int i = 0; i < numSamples; i++){
            x[i] = limits[0] + stepSize * i;
            y[i] = Math.sin(x[i]);
        }
    }

    private void setBestScore(Double bestScore) {
        this.bestScore = Optional.ofNullable(bestScore);
    }

    private SimpleNeuralNetwork getBestNetwork(){
        return trainer.getPopulation().getLeaderBoard((a, b) -> a.compareTo(b)).get(0);
    }

    private String getStatus(){
        StringBuilder update = new StringBuilder();
        try {
        update.append("Status update - ").append(DATE_TIME_FORMAT.format(LocalTime.now()))
              .append("\n\tTime elapsed: ").append(formatTime(Duration.between(startTime, Instant.now())))
              .append("\n\tTrainer status: ").append(trainingThread.isAlive() ? (trainingThread.isInterrupted() ? "Interrupted" : "Active") : "Dead")
              .append("\n\tNetwork Generation: ").append(trainer.getGeneration())
              .append("\n\tNetwork Error: ").append(bestScore.isPresent() ? DECIMAL_FORMAT.format(bestScore.get()) : "N/A")
              .append("\n");
            
        } catch (Exception e) {
            e.printStackTrace();
        }
        return update.toString();
    }

    static void printNetworkPredictions(String networkPath, FileType fileType, double[] x, double[] y) throws FileNotFoundException, IOException, JsonParseException{
        switch (fileType) {
            case JSON:
                printNetworkPredictions(TestingEnvironment.networkFromJson(networkPath, SimpleNeuralNetwork.class), x, y);
                break;
            case SNN:
                printNetworkPredictions(NeuralNetworkBuilder.loadBinary(networkPath + ".snn", new SimpleNeuralNetworkBuilder()).build(), x, y);
                break;
        }
        
    }

    static void printNetworkPredictions(SimpleNeuralNetwork network, double[] x, double[] y){
        try{
            StringBuilder results1 = new StringBuilder();
            for(int i = 0; i < x.length; i++){
                network.setInput(0, x[i]);
                network.forwardPass();
                results1.append("sin(")
                    .append(x[i])
                    .append(") = ")
                    .append(y[i])
                    .append(" -> network(")
                    .append(x[i])
                    .append(") = ")
                    .append(network.getOutput(0))
                    .append("\n");
            }
            println(results1.toString());
            
        } catch (Exception e){
            e.printStackTrace();
        }
    }

    private class InputWatcher implements Runnable{
        final private Scanner INPUT_SCANNER = new Scanner(CONSOLE.reader());

        public void run(){
            if(!INPUT_SCANNER.hasNextLine()) return;

            String input = INPUT_SCANNER.nextLine();
            if(input == null){
                mainThread.interrupt();
                return;
            }

            String[] args = input.split(" ");

            switch (args[0].toLowerCase()) {

                case "save":
                    FileType fileType;
                    if (args.length == 1)
                        fileType = saveType;
                    else
                        try{
                            if (args.length == 2)
                                fileType = FileType.valueOf(args[1].toUpperCase());
                            else
                                throw new Exception("Too many arguments.\nExpected 2, but received " + args.length + ".");
                        } catch(Exception e) {
                            println("Err: " + e.getMessage());
                            break;
                        }
                    try {
                        saveNetwork(getBestNetwork(), savePath, fileType);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    break;

                case "exit", "close", "quit":
                    EXECUTOR_SERVICE.shutdown();
                    System.exit(0);
                    return;

                case "stop", "halt":
                    trainer.stop();
                    synchronized(SYNCH_OBJECT){
                        SYNCH_OBJECT.notifyAll();
                    }
                    return;

                case "view", "print":
                    println(getBestNetwork().copy().toJson());
                    return;

            
                default:
                    println(getStatus());
                    break;
            }
        }
    }

    private class GraphUpdater implements Runnable{
        Optional<Double> previousBestScore = Optional.empty();

        @Override
        public void run() {
            if(bestScore.isEmpty()) return;

            if(previousBestScore.isPresent()){
                if(bestScore.get() >= previousBestScore.get()) return;
            }

            List<RankedNeuralNetwork> leaderBoard = trainer.getLeaderBoard();

            RankedNeuralNetwork bestNetwork = leaderBoard.get(leaderBoard.size() - 1).copy();
            double[] points = Arrays.stream(x)
                    .sequential()
                    .map(x -> {
                        bestNetwork.setInput(0, x);
                        bestNetwork.forwardPass();
                        return bestNetwork.getOutput(0);
                    })
                    .toArray();

            chart.updateXYSeries("Prediction", x, points, null);
            chartWrapper.repaintChart();
        }

    }

    private class StatusUpdater implements Runnable {
        final private Consumer<Double> UPDATE_SCORE = x -> setBestScore(x);

        public synchronized void run(){
            Optional<Double> currentBestScore = trainer.getPopulation().getBestScore();

            if(currentBestScore.isEmpty()) return;

            double leastError = currentBestScore.get();
            if(!Double.isFinite(leastError)) return;

            if(leastError <= acceptableError){
                if(trainingThread.isAlive()){
                    trainer.stop();
                    synchronized(SYNCH_OBJECT){
                        SYNCH_OBJECT.notifyAll();
                    }
                }
            }

            if(bestScore.isEmpty() || (bestScore.get() > leastError)){
                currentBestScore.ifPresent(UPDATE_SCORE);
                println(getStatus());
            }
        }
    }

    private static class ValueMappingTrainer implements TrainingScenario<RankedNeuralNetwork> {
        private ExecutorService executorService = Executors.newCachedThreadPool();
        private ArrayList<RankedNeuralNetwork> networks = new ArrayList<>();
        private ArrayList<Callable<?>> tasks = new ArrayList<>();
        private double[][] predictions;
        private final double[] x, y;

        // static double[] durations = new double[100];
        // static int durationIndex = 0;

        public ValueMappingTrainer(double[] inputs, double[] outputs) throws IllegalArgumentException {
            if(inputs.length != outputs.length)
                throw new IllegalArgumentException("Input and output arrays are differing lengths. (" + inputs.length + " != " + outputs.length + ")");

            x = inputs;
            y = outputs;
        }

        @Override
        public void run() {
            // long start = System.nanoTime();
            predictions = new double[networks.size()][x.length];
            tasks.clear();

            for(int networkIndex = 0; networkIndex < networks.size(); networkIndex++){
                final RankedNeuralNetwork NETWORK = networks.get(networkIndex);
                final double[] PREDICTIONS = predictions[networkIndex];
                tasks.add(() -> { mapValues(NETWORK, x, PREDICTIONS); return null; });
            }

            try {
                executorService.invokeAll(tasks);
            } catch (RejectedExecutionException e) {
                e.printStackTrace();
            } catch (InterruptedException e){}

            // durations[durationIndex++] = System.nanoTime() - start;
            // if (durationIndex == durations.length){
            //     println(((Arrays.stream(durations).sum() / durations.length) / 1000000) + "ms" );
            //     durationIndex = 0;
            // }

            if(frameTime > 0)
                try {
                    Thread.sleep(frameTime);
                } catch (Exception e) {}
        }


        private void mapValues(SimpleNeuralNetwork network, double[] inputs, double[] outputs){
            for(int i = 0; i < x.length; i++){
                network.setInput(0, x[i]);
                network.forwardPass();
                outputs[i] = network.getOutput(0);
            }
        }

        @Override
        public void setParticipants(Collection<RankedNeuralNetwork> c) {
            networks.clear();
            networks.addAll(c);
        }

        @Override
        public void evaluateParticipants() {
            for(int networkIndex = 0; networkIndex < predictions.length; networkIndex++){
                RankedNeuralNetwork network = networks.get(networkIndex);
                double error = LossFunctions.meanSquaredError(predictions[networkIndex], y);

                if(!Double.isFinite(error)){
                    println("TRAINER ENCOUNTERED AN ERROR.\nInvalid totalError: " + error + "\n\nNetwork Layers:\n" + arrayToString(predictions[networkIndex]) + "\n\nOffending Network:\n" + network.toString() + "\n\nAs JSON:\n" + network.toJson() + "\n");
                    System.exit(1);
                }
                
                network.setScore(Double.valueOf(error));
            }
        }
        
    }
}


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
import java.util.Objects;
import java.util.Optional;
import java.util.Scanner;
import java.util.PrimitiveIterator.OfDouble;
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
import com.mjsd.simpleneuralnetwork.NetworkLayout;
import com.mjsd.simpleneuralnetwork.NetworkLayoutBuilder;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.evolution.TrainingScenario;
import com.mjsd.simpleneuralnetwork.training.RankedNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.evolution.SimpleEvolutionaryTrainer;

final public class LearnSineFunction extends TestingEnvironment {
    
    //Settings
    private static String savePath = "SinNetwork.json";
    private static int networksPerGeneration = 500;

    private static int numSamples = 128;
    private static float acceptableError = 0.0005f;
    private static double[] limits = new double[]{0, 2 * Math.PI};

    private static NetworkLayout layout = new NetworkLayoutBuilder()
                                          .withInputLayer(1)
                                          .withOutputLayer(1)
                                          .addLayers(2, 4, ActivationFunctions.TANH)
                                          .build();

    //Class
    final private Console CONSOLE;
    private XYChart chart;
    private SwingWrapper<XYChart> chartWrapper;

    final private Object SYNCH_OBJECT = new Object();
    final private ScheduledExecutorService EXECUTOR_SERVICE = Executors.newScheduledThreadPool(3);
    final private DecimalFormat DECIMAL_FORMAT = new DecimalFormat("0.00##");
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
        println("Network Layout:\n" + layout);

        try{ printNetworkPredictions(savePath, x, y); } catch (Exception e){}

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

        trainer = new SimpleEvolutionaryTrainer<>(networksPerGeneration, () -> new RankedNeuralNetwork(layout), (a, b) -> 0 - a.compareTo(b), new ValueMappingTrainer(x, y));
        trainer.getEcosystem().setParallel(true);

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

        Optional<Double> bestScore = trainer.getEcosystem().getBestScore();
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
        return trainer.getEcosystem().getLeaderBoard((a, b) -> a.compareTo(b)).get(0);
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

    static void printNetworkPredictions(String networkPath, double[] x, double[] y) throws FileNotFoundException, IOException, JsonParseException{
        printNetworkPredictions(TestingEnvironment.networkFromFile(networkPath, SimpleNeuralNetwork.class), x, y);
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

            switch (input.toLowerCase()) {

                case "save":
                    try {
                        saveNetwork(getBestNetwork(), savePath);
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

            RankedNeuralNetwork bestNetwork = trainer.getLeaderBoard().get(0);
            double[] points = new double[x.length];
            OfDouble yValues = Arrays.stream(x)
                    .map(x -> {
                        bestNetwork.setInput(0, x);
                        bestNetwork.forwardPass();
                        return bestNetwork.getOutput(0);
                    })
                    .iterator();
            for(int i = 0; i < points.length; i++)
                points[i] = yValues.next();

            chart.updateXYSeries("Prediction", x, points, null);
            chartWrapper.repaintChart();
        }

    }

    private class StatusUpdater implements Runnable {
        final private Consumer<Double> UPDATE_SCORE = x -> setBestScore(x);

        public synchronized void run(){
            Optional<Double> currentBestScore = trainer.getEcosystem().getBestScore();

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

        public ValueMappingTrainer(double[] inputs, double[] outputs) throws IllegalArgumentException {
            if(inputs.length != outputs.length)
                throw new IllegalArgumentException("Input and output arrays are differing lengths. (" + inputs.length + " != " + outputs.length + ")");

            x = inputs;
            y = outputs;
        }

        @Override
        public void run() {
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
                double totalError = 0;
                for(int predictionIndex = 0; predictionIndex < predictions[networkIndex].length; predictionIndex++){
                    totalError += Math.pow(predictions[networkIndex][predictionIndex] - y[predictionIndex], 2);
                }

                if(!Double.isFinite(totalError)){
                    println("TRAINER ENCOUNTERED AN ERROR.\nInvalid totalError: " + totalError + "\n\nNetwork Layers:\n" + arrayToString(predictions[networkIndex]) + "\n\nOffending Network:\n" + network.toString() + "\n\nAs JSON:\n" + network.toJson() + "\n");
                    System.exit(1);
                }
                
                network.setScore(Double.valueOf(totalError));
            }
        }
        
    }
}


import java.io.Console;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.DecimalFormat;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.Scanner;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.AxesChartStyler;
import org.knowm.xchart.style.theme.GGPlot2Theme;

import com.github.thecybrix.simpleneuralnetwork.core.ActivationFunctions;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetworkBuilder;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkBuilder;
import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.ScoredNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelector;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ValueMappingTrainer;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.simple.SimpleEvolutionaryTrainer;
import com.google.gson.JsonParseException;

final public class LearnSineFunction extends TestingEnvironment {
    
    //Settings
    private static boolean continueTraining = false;
    private static float fpsLimit = 0.0f;
    private static String savePath = "TestSaves\\SinNetwork";
    private static FileType saveType = FileType.JSON;
    private static int networksPerGeneration = 100;

    private static int numSamples = 128;
    private static float acceptableError = 0.000005f;
    private static double[] limits = new double[]{0, 2 * Math.PI};

    private static NeuralNetworkBuilder<MutableNeuralNetwork> layout = new MutableNeuralNetworkBuilder()
                                                                      .withInputLayer(1)
                                                                      .withOutputLayer(1)
                                                                      .addHiddenLayers(2, 4, ActivationFunctions.TANH);

    //Class
    final private Console CONSOLE;
    private XYChart chart;
    private SwingWrapper<XYChart> chartWrapper;


    private int numXTicks = 5;
    private int numYTicks = 5;

    private static long frameTime = fpsLimit <= 0 ? 0l : Math.round(1000 / fpsLimit);

    final private Object SYNCH_OBJECT = new Object();
    final private ScheduledExecutorService EXECUTOR_SERVICE = Executors.newScheduledThreadPool(3);
    final private DecimalFormat DECIMAL_FORMAT = new DecimalFormat("0.00#####");
    final private DateTimeFormatter DATE_TIME_FORMAT = DateTimeFormatter.ofPattern("h':'mm a");
    
    private Thread mainThread, trainingThread;
    private SimpleEvolutionaryTrainer<MutableNeuralNetwork> trainer;

    private Instant startTime, endTime;
    private OptionalDouble bestScore = OptionalDouble.empty();
    private Optional<MutableNeuralNetwork> bestNetwork = Optional.empty();

    final double[][] inputs, outputs;
    final double[] x, y;
    final double[] error;

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

        chart.getStyler().setTheme(new GGPlot2Theme());
        // Customize styling for a dark theme
        AxesChartStyler styler = (AxesChartStyler) chart.getStyler();
        styler.setChartBackgroundColor(java.awt.Color.BLACK);
        styler.setPlotBackgroundColor(java.awt.Color.DARK_GRAY);
        styler.setChartFontColor(java.awt.Color.WHITE);
        styler.setAxisTickLabelsColor(java.awt.Color.WHITE);
        styler.setXAxisTitleColor(java.awt.Color.LIGHT_GRAY);
        styler.setYAxisTitleColor(java.awt.Color.LIGHT_GRAY);
        styler.setLegendBackgroundColor(java.awt.Color.DARK_GRAY);
        styler.setLegendBorderColor(java.awt.Color.GRAY);
        styler.setPlotGridLinesColor(java.awt.Color.GRAY);
        styler.setChartTitleBoxBackgroundColor(java.awt.Color.BLACK);
        styler.setChartTitleBoxBorderColor(java.awt.Color.GRAY);
        styler.setPlotGridLinesVisible(true);

        styler.setMarkerSize(0);
        // styler.setYAxisTicksVisible(false);

        int xSpacingHint = (int) ((chart.getWidth() / numXTicks) * 0.5);
        int ySpacingHint = (int) ((chart.getHeight() / numYTicks) * 0.5);
        styler.setXAxisTickMarkSpacingHint(xSpacingHint);
        styler.setYAxisTickMarkSpacingHint(ySpacingHint);

        styler.setXAxisMin(limits[0] - 0.1);
        styler.setXAxisMax(limits[1] + 0.1);
        styler.setYAxisMin(-1.1);
        styler.setYAxisMax(1.1);


        chart.addSeries("Sine", x, y, null);
        chart.addSeries("Prediction", x, new double[x.length], null);
        chartWrapper = new SwingWrapper<>(chart);
        chartWrapper.displayChart();

        trainer = new SimpleEvolutionaryTrainer<>(networksPerGeneration, layout::build, ParentSelector.eliteSelection(Comparator.reverseOrder()), ValueMappingTrainer.of(inputs, outputs, new ValueMappingTrainer.MeanSquaredError()), (a, b) -> 0 - a.compareTo(b));
        trainer.attachCallback(x -> updateScoreHistory());
        if(fpsLimit > 0) trainer.attachCallback(x -> { try { Thread.sleep(frameTime); } catch (Exception e) {}; });

        if(continueTraining){
            try {
                MutableNeuralNetworkBuilder builder = new MutableNeuralNetworkBuilder(loadNetworkFromFile(savePath, saveType));
                for (int i = 0; i < networksPerGeneration; i++)
                    trainer.addNetwork(builder.build());
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(1);
            }
        }
        
        trainingThread = new Thread(trainer);
        // trainingThread.start();

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

        OptionalDouble bestScore = Collections.min(trainer.getPreviousGeneration()).getScore();
        SimpleNeuralNetwork bestNetwork = getBestNetwork().get();

        results.append((bestScore.getAsDouble() <= acceptableError) ? "Network reached desired proficiency in " : "Training stopped after ")
               .append(formatTime(Duration.between(startTime, endTime)))
               .append(".\nAverage error: ")
               .append((bestScore.isPresent() ? DECIMAL_FORMAT.format(bestScore.getAsDouble()) : "< " + acceptableError))
               .append("\nNetwork Generation: ")
               .append(trainer.getGeneration())
               .append("\n\n");

        for(int i = 0; i < inputs.length; i++){
            bestNetwork.setInput(0, inputs[i][0]);
            bestNetwork.forwardPass();
            results.append(", ")
                   .append(DECIMAL_FORMAT.format(inputs[i][0]))
                   .append(", ")
                   .append(DECIMAL_FORMAT.format(outputs[i][0]))
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
        inputs = new double[numSamples][1]; 
        outputs = new double[numSamples][1];
        error = new double[numSamples];

        double stepSize = range / (double)numSamples;
        for(int i = 0; i < numSamples; i++){
            x[i] = limits[0] + stepSize * i;
            y[i] = Math.sin(x[i]);
            inputs[i][0] = x[i];
            outputs[i][0] = y[i];
        }
    }

    private void updateScoreHistory(){
        List<ScoredNetwork<MutableNeuralNetwork>> prevGen = trainer.getPreviousGeneration();
        if(prevGen.isEmpty()) return;
        ScoredNetwork<MutableNeuralNetwork> bestScoredNetwork = prevGen.parallelStream().filter(x -> x.getScore().isPresent()).min((x, y) -> x.compareTo(y)).orElse(null);
        if(bestScoredNetwork == null) return;
        OptionalDouble newBestScore = bestScoredNetwork.getScore();
        if(newBestScore.isEmpty()) return;
        if(bestScore.isEmpty() || newBestScore.getAsDouble() < bestScore.getAsDouble()){
            bestNetwork = Optional.ofNullable(bestScoredNetwork.get().copy());
            bestScore = newBestScore;
        }
    }

    private OptionalDouble getBestScore() {
        return bestScore;
    }

    private Optional<MutableNeuralNetwork> getBestNetwork(){
        return bestNetwork;
    }

    private String getStatus(){
        StringBuilder update = new StringBuilder();
        try {
        update.append("Status update - ").append(DATE_TIME_FORMAT.format(LocalTime.now()))
              .append("\n\tTime elapsed: ").append(formatTime(Duration.between(startTime, Instant.now())))
              .append("\n\tTrainer status: ").append(trainingThread.isAlive() ? (trainingThread.isInterrupted() ? "Interrupted" : "Active") : "Dead")
              .append("\n\tNetwork Generation: ").append(trainer.getGeneration())
              .append("\n\tNetwork Error: ").append(bestScore.isPresent() ? DECIMAL_FORMAT.format(bestScore.getAsDouble()) : "N/A")
              .append("\n");
            
        } catch (Exception e) {
            e.printStackTrace();
        }
        return update.toString();
    }

    static void printNetworkPredictions(String networkPath, FileType fileType, double[] x, double[] y) throws FileNotFoundException, IOException, JsonParseException{
        printNetworkPredictions(loadNetworkFromFile(networkPath, fileType), x, y);
    }

    static MutableNeuralNetwork loadNetworkFromFile(String networkPath, FileType fileType) throws FileNotFoundException, IOException, JsonParseException{
        switch (fileType) {
            case JSON:
                return TestingEnvironment.networkFromJson(networkPath, MutableNeuralNetwork.class);
            case SNN:
                return NeuralNetworkBuilder.loadBinary(networkPath + ".snn", new MutableNeuralNetworkBuilder()).build();
            default:
                throw new FileNotFoundException();
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

                case "start":
                    if(!trainingThread.isAlive())
                        trainingThread.start();
                    return;

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
                        MutableNeuralNetwork bestNetwork = getBestNetwork().orElse(null);
                        if(bestNetwork != null) saveNetwork(bestNetwork, savePath, fileType);
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
                    println(getBestNetwork().get().copy().toJson());
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
            if(getBestScore().isEmpty()) return;

            double currentBestScore = getBestScore().getAsDouble();

            if(previousBestScore.isPresent()){
                if(currentBestScore >= previousBestScore.get()) return;
            }

            
            MutableNeuralNetwork bestNetwork = getBestNetwork().orElse(null);
            if(bestNetwork == null) return;
            
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
        private Optional<Double> previousBestScore = Optional.empty();

        public synchronized void run(){
            if(getBestScore().isEmpty()) return;
            double currentBestScore = getBestScore().getAsDouble();

            if(!Double.isFinite(currentBestScore)) return;

            if(currentBestScore <= acceptableError){
                if(trainingThread.isAlive()){
                    trainer.stop();
                    synchronized(SYNCH_OBJECT){
                        SYNCH_OBJECT.notifyAll();
                    }
                }
            }

            if(previousBestScore.isEmpty() || (previousBestScore.get() > currentBestScore)){
                println(getStatus());
                previousBestScore = Optional.of(currentBestScore);
            }
        }
    }
    
}

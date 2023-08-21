
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.Console;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Optional;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import com.google.gson.Gson;
import com.google.gson.JsonParseException;
import com.mjsd.simpleneuralnetwork.ActivationFunctions;
import com.mjsd.simpleneuralnetwork.NetworkLayout;
import com.mjsd.simpleneuralnetwork.NetworkLayoutBuilder;
import com.mjsd.simpleneuralnetwork.NeuralNetworkBuilder;
import com.mjsd.simpleneuralnetwork.NeuralNetworkTools;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork;
import com.mjsd.simpleneuralnetwork.gson.CustomGsonFactory;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetworkBuilder;
import com.mjsd.simpleneuralnetwork.training.evolution.TrainingScenario;
import com.mjsd.simpleneuralnetwork.training.RankedNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.evolution.SimpleEvolutionaryTrainer;

public class Test {

    final static Gson GSON = CustomGsonFactory.getInstance().newBuilder().setPrettyPrinting().create();
    static boolean saveNetwork = false;

    public static void main(String[] args) {
        String savePath = "SinNetwork.json";

        NetworkLayout layout = new NetworkLayoutBuilder()
                               .withInputLayer(1)
                               .withOutputLayer(1)
                               .addLayers(2, 4, ActivationFunctions.TANH)
                               .build();
        
        int numSamples = 128;
        float acceptableError = 0.02f;
        double[] limits = new double[]{0, 2 * Math.PI};
        double range = limits[1] - limits[0];
        final double[] x = new double[numSamples],
                       y = new double[numSamples];

        double stepSize = range / (double)numSamples;
        for(int i = 0; i < numSamples; i++){
            x[i] = limits[0] + stepSize * i;
            y[i] = Math.sin(x[i]);
        }

        println(arrayToString(y));

        try{
            printNetworkPredictions(savePath, x, y);
        } catch (Exception e){}

        TrainingScenario<RankedNeuralNetwork> learnSinFunction = new TrainingScenario<>() {

            ArrayList<RankedNeuralNetwork> networks = new ArrayList<>();
            double[][] predictions;

            @Override
            public void run() {

                predictions = new double[networks.size()][numSamples];

                for(int networkIndex = 0; networkIndex < networks.size(); networkIndex++){
                    RankedNeuralNetwork network = networks.get(networkIndex);
                    for(int input = 0; input < x.length; input++){
                        network.setInput(0, x[input]);
                        network.forwardPass();
                        predictions[networkIndex][input] = network.getOutput(0);
                    }
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
                        println( "what??? " + totalError + "\n" + arrayToString(predictions[networkIndex]) + "\n" + network.toString() + "\n");
                        println(network.toJson());
                        System.exit(1);
                    }
                    
                    network.setScore(Double.valueOf(totalError));
                }
            }
            
        };

        SimpleEvolutionaryTrainer<RankedNeuralNetwork> trainer = new SimpleEvolutionaryTrainer<>(500, () -> new RankedNeuralNetwork(layout), (a, b) -> 0 - a.compareTo(b), learnSinFunction);
        trainer.getEcosystem().setParallel(true);

        println(layout);

        Thread trainingThread = new Thread(trainer);

        Console console = System.console();
        if(console == null) return;

        DecimalFormat decimalFormat = new DecimalFormat("0.00##");
        Object waitObject = new Object();

        trainingThread.start();
        ScheduledExecutorService statusUpdateService = Executors.newSingleThreadScheduledExecutor();
        statusUpdateService.scheduleAtFixedRate(
            new Runnable() {
                Optional<Double> bestScore = Optional.empty();
                public void run(){
                Optional<Double> currentBestScore = trainer.getEcosystem().getBestScore();

                if(currentBestScore.isEmpty()) return;

                double leastError = currentBestScore.get();
                if(!Double.isFinite(leastError)) return;

                if(leastError <= acceptableError){
                    if(trainingThread.isAlive()){
                        trainer.stop();
                        synchronized(waitObject){
                            waitObject.notifyAll();
                        }
                    }
                }

                if(bestScore.isEmpty() || (bestScore.get().doubleValue() > leastError)){
                    bestScore = currentBestScore;
                    println("\nTrainer status: " + (trainingThread.isAlive() ? (trainingThread.isInterrupted() ? "Interrupted" : "Active") : "Dead") + "\nNetwork Generation: " + trainer.getGeneration() + "\nNetwork Error: " + (bestScore.isPresent() ? decimalFormat.format(bestScore.get()) : "N/A"));
                }
            }
        }, 50, 250, TimeUnit.MILLISECONDS);

        Thread mainThread = Thread.currentThread();
        Thread inputReader = new Thread(new Runnable(){
            public void run(){
                while(trainingThread.isAlive()){
                    String input = console.readLine();
                    if(input == null){
                        mainThread.interrupt();
                        return;
                    }

                    switch (input.toLowerCase()) {

                        case "save":
                            saveNetwork = true;

                        case "exit", "close", "quit":
                            trainer.stop();
                            synchronized(waitObject){
                                waitObject.notifyAll();
                            }
                            return;
                            
                    
                        default:
                            println("Trainer status: " + (trainingThread.isAlive() ? (trainingThread.isInterrupted() ? "Interrupted" : "Active") : "Dead"));
                            println("Num networks = " + trainer.getEcosystem().getCurrentGeneration().size());
                            break;
                    }
                }
            }
        });
        inputReader.start();

        try {
            synchronized(waitObject){
                waitObject.wait();
            }
        } catch (Exception exception) {
            if(!(exception instanceof InterruptedException)) exception.printStackTrace();
            println("Closing application...");
            trainer.stop();
            return;
        } finally {
            statusUpdateService.shutdown();
        }

        Optional<Double> bestScore = trainer.getEcosystem().getBestScore();
        println("\n\nNetwork reached desired proficiency.\nAverage error of " + decimalFormat.format(acceptableError) + ".\n");
        println("\nNetwork Generation: " + trainer.getGeneration() + "\nNetwork Error: " + (bestScore.isPresent() ? decimalFormat.format(bestScore.get()) : "N/A"));

        StringBuilder results = new StringBuilder();
        SimpleNeuralNetwork bestNetwork = trainer.getEcosystem().getLeaderBoard((a, b) -> a.compareTo(b)).get(0);
        for(int i = 0; i < x.length; i++){
            bestNetwork.setInput(0, x[i]);
            bestNetwork.forwardPass();
            //results.append("sin(");
            results.append(", ");
            results.append(decimalFormat.format(x[i]));
            //results.append(") = ");
            results.append(", ");
            results.append(decimalFormat.format(y[i]));
            //results.append(" ~ ");
            results.append(", ");
            results.append(decimalFormat.format(bestNetwork.getOutput(0)));
            //results.append("\n");
        }
        println(results.toString());

        if(saveNetwork) saveNetwork(bestNetwork, savePath, null);

        System.exit(0);

    }


    static String arrayToString(double[] array){
        StringBuilder out = new StringBuilder();
        
        out.append("[").append(array[0]);
        for(int i = 1; i < array.length; i++)
            out.append(", ").append(array[i]);
        out.append("]");

        return out.toString();
    }

    static void jsonParserTest(MutableNeuralNetwork originalNetwork){
        MutableNeuralNetwork serializedNetwork;
        String asJson;
        asJson = originalNetwork.toJson(GSON);

        double[] inputValues = new double[originalNetwork.getInputLayer().length];
        for(int i = 0; i < inputValues.length; i++)
            inputValues[i] = Math.random();

        NeuralNetworkTools.randomizeWeightsAndBiases(originalNetwork, 10, 10);
        originalNetwork.setInput(inputValues);
        originalNetwork.forwardPass();


        serializedNetwork = MutableNeuralNetworkBuilder.fromJson(asJson);
        println(serializedNetwork.getLayout() + "\n\n");

        println(asJson);
        println("\n\nShould be false: " + originalNetwork.equals(serializedNetwork));

        asJson = originalNetwork.toJson(GSON);
        serializedNetwork = MutableNeuralNetworkBuilder.fromJson(asJson);

        println(asJson);
        println("\n\nShould be true: " + originalNetwork.equals(serializedNetwork));
    }

    static void println(String string){
        System.out.println(string);
    }

    static void println(Object obj){
        println(obj.toString());
    }

    static <T extends SimpleNeuralNetwork> T networkFromFile(String filePath, Class<T> networkType) throws FileNotFoundException, IOException, JsonParseException {
        File saveFile = new File(filePath);

        if(!saveFile.isFile()) throw new FileNotFoundException("File does not exist: " + saveFile.getAbsolutePath());

        BufferedReader reader = new BufferedReader(new FileReader(saveFile));
        StringBuffer buffer = new StringBuffer();
        String fragment = reader.readLine();
        while(fragment != null){
            buffer.append(fragment);
            fragment = reader.readLine();
        }
        reader.close();
        return NeuralNetworkBuilder.fromJson(buffer.toString(), networkType);
    }

    static void printNetworkPredictions(String networkPath, double[] x, double[] y) throws FileNotFoundException, IOException, JsonParseException{
        printNetworkPredictions(networkFromFile(networkPath, SimpleNeuralNetwork.class), x, y);
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
                    .append(" ~ ")
                    .append(network.getOutput(0))
                    .append("\n");
            }
            println(results1.toString());
            
        } catch (Exception e){
            e.printStackTrace();
        }
    }

    static void saveNetwork(SimpleNeuralNetwork network, String path) throws IOException{
        saveNetwork(network, path, null);
    }

    static void saveNetwork(SimpleNeuralNetwork network, String path, Consumer<Exception> onException) {
        onException = (onException == null) ? x -> x.printStackTrace() : onException;
        try {
            File saveFile = new File(path);
            if(!saveFile.exists()) saveFile.createNewFile();
            BufferedWriter writer = new BufferedWriter(new FileWriter(saveFile));
            writer.write(network.toJson());
            writer.flush();
            writer.close();
        } catch (Exception e) {
            onException.accept(e);
        }
    }
}

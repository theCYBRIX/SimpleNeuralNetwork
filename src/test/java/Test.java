
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.Console;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

import com.google.gson.Gson;

import com.mjsd.simpleneuralnetwork.ActivationFunctions;
import com.mjsd.simpleneuralnetwork.NetworkLayout;
import com.mjsd.simpleneuralnetwork.NetworkLayoutBuilder;
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

    public static void main(String[] args) {
        String saveLocation = "SinNetwork.json";
        File saveFile;
        /*
        AdjustableNeuralNetwork originalNetwork;

        AdjustableNeuralNetworkBuilder networkBuilder = new AdjustableNeuralNetworkBuilder();
        networkBuilder.withInputLayer(2, InputNormalizers.NO_NORMALIZER)
                      .addHiddenLayer(3, ActivationFunctions.LINEAR)
                      .withOutputLayer(1, ActivationFunctions.LINEAR)
                      .withWeights(new double[][][]{{{ -0.84391201, 0.2454178}, {-0.47772055, -0.80820501}, {-0.94225327, 0.55239087}}, {{-0.28487789, 0.78176581, -0.2472606}}});;

        originalNetwork = networkBuilder.build();
        */

        
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

        System.out.println(arrayToString(y));

        /*
        saveFile = new File(saveLocation);
        if(saveFile.isFile()){
            try{
                BufferedReader reader = new BufferedReader(new FileReader(saveFile));
                StringBuffer buffer = new StringBuffer();
                String fragment = reader.readLine();
                while(fragment != null){
                    buffer.append(fragment);
                    fragment = reader.readLine();
                }
                reader.close();
                AdjustableNeuralNetwork network = AdjustableNeuralNetworkBuilder.fromJson(buffer.toString());

                

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
                System.out.println(results1.toString());
                
            } catch (Exception e){
                e.printStackTrace();
            }

            return;
        }
        */

        com.mjsd.simpleneuralnetwork.training.evolution.TrainingScenario<RankedNeuralNetwork> learnSinFunction = new TrainingScenario<>() {

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
                /*
                try {
                    Thread.sleep(1);
                } catch (Exception e) {
                    Thread.currentThread().interrupt();
                }
                */
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
                        System.out.println( "what??? " + totalError + "\n" + arrayToString(predictions[networkIndex]) + "\n" + network.toString() + "\n");
                        System.out.println(network.toJson());
                        System.exit(1);
                    }
                    
                    network.setScore(Double.valueOf(totalError));
                }
            }
            
        };

        NetworkLayout layout = new NetworkLayoutBuilder()
                               .withInputLayer(1)
                               .withOutputLayer(1)
                               .addLayers(10, 8, ActivationFunctions.ReLU)
                               .build();

        SimpleEvolutionaryTrainer<RankedNeuralNetwork> trainer = new SimpleEvolutionaryTrainer<>(500, () -> new RankedNeuralNetwork(layout), (a, b) -> 0 - a.compareTo(b), learnSinFunction);
        trainer.getEcosystem().setParallel(true);

        System.out.println(layout);
        trainer.getEcosystem().populateNewGeneration();
        System.out.println("Num networks = " + trainer.getEcosystem().getCurrentGeneration().size());

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
                    System.out.println("\nTrainer status: " + (trainingThread.isAlive() ? (trainingThread.isInterrupted() ? "Interrupted" : "Active") : "Dead") + "\nNetwork Generation: " + trainer.getGeneration() + "\nNetwork Error: " + (bestScore.isPresent() ? decimalFormat.format(bestScore.get()) : "N/A"));
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
                        case "exit", "close", "quit":
                            mainThread.interrupt();
                            return;

                        case "save":
                            trainer.stop();
                            synchronized(waitObject){
                                waitObject.notifyAll();
                            }
                            return;
                            
                    
                        default:
                            System.out.println("Trainer status: " + (trainingThread.isAlive() ? (trainingThread.isInterrupted() ? "Interrupted" : "Active") : "Dead"));
                            System.out.println("Num networks = " + trainer.getEcosystem().getCurrentGeneration().size());
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
            System.out.println("Closing application...");
            trainer.stop();
            return;
        } finally {
            statusUpdateService.shutdown();
        }

        Optional<Double> bestScore = trainer.getEcosystem().getBestScore();
        System.out.println("\n\nNetwork reached desired proficiency.\nAverage error of " + decimalFormat.format(acceptableError) + ".\n");
        System.out.println("\nNetwork Generation: " + trainer.getGeneration() + "\nNetwork Error: " + (bestScore.isPresent() ? decimalFormat.format(bestScore.get()) : "N/A"));

        StringBuilder results = new StringBuilder();
        SimpleNeuralNetwork bestNetwork = trainer.getEcosystem().getCurrentGeneration().get(0); //TODO: fix
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
        System.out.println(results.toString());

        try{
            saveFile = new File(saveLocation);
            if(!saveFile.exists()) saveFile.createNewFile();
            BufferedWriter writer = new BufferedWriter(new FileWriter(saveFile));
            writer.write(bestNetwork.toJson());
            writer.flush();
            writer.close();
        } catch(Exception e){
            e.printStackTrace();
        }

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
        System.out.println(serializedNetwork.getLayout() + "\n\n");

        System.out.println(asJson);
        System.out.println("\n\nShould be false: " + originalNetwork.equals(serializedNetwork));

        asJson = originalNetwork.toJson(GSON);
        serializedNetwork = MutableNeuralNetworkBuilder.fromJson(asJson);

        System.out.println(asJson);
        System.out.println("\n\nShould be true: " + originalNetwork.equals(serializedNetwork));
    }
}

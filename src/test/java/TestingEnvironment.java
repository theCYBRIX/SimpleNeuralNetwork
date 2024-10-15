import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.time.Duration;
import java.util.function.Consumer;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkBuilder;
import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.serialization.binary.NetworkSerializer;
import com.google.gson.JsonParseException;

public abstract class TestingEnvironment {

    public enum FileType {
        SNN,
        JSON
    }

    static void println(String string){
        System.out.println(string);
    }

    static void println(Object obj){
        println(obj.toString());
    }

    public static String arrayToString(double[] array){
        StringBuilder out = new StringBuilder();
        
        out.append("[").append(array[0]);
        for(int i = 1; i < array.length; i++)
            out.append(", ").append(array[i]);
        out.append("]");

        return out.toString();
    }

    public static String networkToString(MutableNeuralNetwork network){
        StringBuilder builder = new StringBuilder();
        
        double[][][] weights = network.getWeights();
        double[][] biases = network.getBiases();

        for (int l = 0; l < weights.length; l++) {
            builder.append("{Layer ").append(l).append("}\n");
            for (int n = 0; n < weights[l].length; n++) {
            builder.append("\t{Node ").append(n).append("}\n");
                builder.append("\t\t{Weights} - [").append(weights[l][n][0]);
                for (int w = 1; w < weights[l][n].length; w++) 
                    builder.append(", ").append(weights[l][n][w]);
                builder.append("]\n\t\t{Bias} - [").append(biases[l][n]).append("]\n");
            }
        }
        
        return builder.toString();
    }

    public static String formatTime(Duration duration){
        StringBuilder formatted = new StringBuilder();
        int hours = duration.toHoursPart(),
            minutes = duration.toMinutesPart(),
            seconds = duration.toSecondsPart();

        if(hours > 0) formatted.append(hours).append("h ");
        if(minutes > 0 || formatted.length() > 0) formatted.append(minutes).append("m ");
        formatted.append(seconds).append("s");

        return formatted.toString(); 
    }

    public static <T extends SimpleNeuralNetwork> T networkFromJson(String filePath, Class<T> networkType) throws FileNotFoundException, IOException, JsonParseException {
        if(!filePath.endsWith(".json")) filePath += ".json";
        File saveFile = new File(filePath);

        if(!saveFile.isFile()) throw new FileNotFoundException("File does not exist: " + saveFile.getAbsolutePath());

        StringBuffer buffer = new StringBuffer();

        try(BufferedReader reader = new BufferedReader(new FileReader(saveFile))){
            String fragment = reader.readLine();
            while(fragment != null){
                buffer.append(fragment);
                fragment = reader.readLine();
            }
        }
        
        return NeuralNetworkBuilder.fromJson(buffer.toString(), networkType);
    }

    public static void saveNetwork(SimpleNeuralNetwork network, String path, FileType fileType) throws IOException{
        saveNetwork(network, path, fileType, null);
    }

    public static void saveNetwork(SimpleNeuralNetwork network, String path, FileType fileType, Consumer<Exception> onException) {

        switch (fileType) {
            case JSON:
                saveNetworkJson(network, path, onException);
                break;
            case SNN:
                saveNetworkSNN(network, path, onException);
                break;
        }

    }

    public static void saveNetworkJson(SimpleNeuralNetwork network, String path, Consumer<Exception> onException){

        if(!path.endsWith(".json")) path += ".json";

        saveToFile(path, network.toJson(), onException);
    }

    public static void saveNetworkSNN(SimpleNeuralNetwork network, String path, Consumer<Exception> onException){
        onException = (onException == null) ? x -> x.printStackTrace() : onException;
        NetworkSerializer serializer = new NetworkSerializer();
        try {
            serializer.save(network, path, true);
        } catch (Exception e) {
            onException.accept(e);
        }
    }

    public static void saveToFile(String filePath, String data, Consumer<Exception> onException) throws NullPointerException{
        if(onException == null) onException = x -> x.printStackTrace();
        try {
            saveToFile(filePath, data);
        } catch (Exception e) {
            onException.accept(e);
        }
    }

    public static void saveToFile(String path, String data) throws IOException, SecurityException, NullPointerException {
        File saveFile = new File(path);
        if(!saveFile.exists()) saveFile.createNewFile();

        try(BufferedWriter writer = new BufferedWriter(new FileWriter(saveFile))) {
            writer.write(data);
        }
    }
}

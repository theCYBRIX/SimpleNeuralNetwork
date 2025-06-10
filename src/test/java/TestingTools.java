import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.time.Duration;
import java.util.IllegalFormatException;
import java.util.function.Consumer;
import java.util.function.DoubleFunction;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkBuilder;
import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.serialization.binary.NetworkSerializer;
import com.google.gson.JsonParseException;

public abstract class TestingTools {
    
    final static String RED = "\u001B[31m";
    final static String GREEN = "\u001B[32m";
    final static String BLUE = "\u001B[34m";
    final static String RESET = "\u001B[0m";

    public enum FileType {
        SNN,
        JSON
    }

    static void print(String string){
        System.out.print(string);
    }

    static void print(Object obj){
        print(obj.toString());
    }

    static void println(){
        System.out.println();
    }

    static void println(String string){
        System.out.println(string);
    }

    static void println(Object obj){
        println(obj.toString());
    }
    
    static void printSideBySide(String label1, String block1, String label2, String block2) {
        String[] lines1 = block1.split("\n");
        String[] lines2 = block2.split("\n");

        int maxContentWidth = Math.max(getMaxWidth(lines1), getMaxWidth(lines2));
        int cleanLabel1Len = stripAnsi(label1).length();
        int cleanLabel2Len = stripAnsi(label2).length();
        int columnWidth = Math.max(Math.max(cleanLabel1Len, cleanLabel2Len), maxContentWidth) + 2;

        String paddedLabel1 = padWithAnsi(label1, columnWidth);
        String paddedLabel2 = padWithAnsi(label2, columnWidth);

        System.out.println(paddedLabel1 + " │ " + paddedLabel2);

        int maxLines = Math.max(lines1.length, lines2.length);
        for (int i = 0; i < maxLines; i++) {
            String left = i < lines1.length ? lines1[i] : "";
            String right = i < lines2.length ? lines2[i] : "";
            System.out.printf("%-" + columnWidth + "s│ %s%n", left, right);
        }
    }

    static int getMaxWidth(String[] lines) {
        int max = 0;
        for (String line : lines) {
            max = Math.max(max, stripAnsi(line).length());
        }
        return max;
    }

    static String stripAnsi(String input) {
        return input.replaceAll("\u001B\\[[;\\d]*m", "");
    }

    static String padWithAnsi(String original, int width) {
        String clean = stripAnsi(original);
        int paddingNeeded = width - clean.length();
        StringBuilder sb = new StringBuilder(original);
        for (int i = 0; i < paddingNeeded; i++) {
            sb.append(" ");
        }
        return sb.toString();
    }

    static String formatBoolean(boolean value){
        return ((value ? GREEN : RED) + value + RESET);
    }

    static String formatColorized(Object obj, String color){
        return formatColorized(obj.toString(), color);
    }

    static String formatColorized(String string, String color){
        return (color + string + RESET);
    }

    public static void addFileLogHandler(Logger logger, String filePath, boolean append) throws IOException{
        FileHandler logFileHandler = new FileHandler(filePath, append);
        logFileHandler.setFormatter(new SimpleFormatter());
        logFileHandler.setLevel(Level.ALL);
        logger.addHandler(logFileHandler);
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
        return networkToString(network, "%.6f");
    }

    public static String networkToString(MutableNeuralNetwork network, String doubleFormat) throws IllegalFormatException {
        StringBuilder builder = new StringBuilder();
        
        double[][][] weights = network.getWeights();
        double[][] biases = network.getBiases();
        DoubleFunction<String> formatDouble = x -> String.format(doubleFormat, x);

        for (int l = 0; l < weights.length; l++) {
            builder.append("Layer ").append(l).append(":\n");
            for (int n = 0; n < weights[l].length; n++) {
            builder.append("  Node ").append(n).append(":\n");
                builder.append("    Weights: [").append(formatDouble.apply(weights[l][n][0]));
                for (int w = 1; w < weights[l][n].length; w++) 
                    builder.append(", ").append(formatDouble.apply(weights[l][n][w]));
                builder.append("]\n    Bias: [").append(formatDouble.apply(biases[l][n])).append("]\n");
            }
            if(l + 1 < weights.length) builder.append("\n");
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

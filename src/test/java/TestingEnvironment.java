import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.time.Duration;
import java.util.function.Consumer;

import com.google.gson.JsonParseException;
import com.mjsd.simpleneuralnetwork.NeuralNetworkBuilder;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork;

public abstract class TestingEnvironment {

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

    public static <T extends SimpleNeuralNetwork> T networkFromFile(String filePath, Class<T> networkType) throws FileNotFoundException, IOException, JsonParseException {
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

    public static void saveNetwork(SimpleNeuralNetwork network, String path) throws IOException{
        saveNetwork(network, path, null);
    }

    public static void saveNetwork(SimpleNeuralNetwork network, String path, Consumer<Exception> onException) {
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

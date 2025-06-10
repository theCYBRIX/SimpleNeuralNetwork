import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;

import com.github.thecybrix.simpleneuralnetwork.core.ActivationFunctions;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetworkBuilder;
import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayout;
import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayoutBuilder;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkTools;
import com.github.thecybrix.simpleneuralnetwork.serialization.binary.NetworkDeserializer;
import com.github.thecybrix.simpleneuralnetwork.serialization.binary.NetworkSerializer;
import com.github.thecybrix.simpleneuralnetwork.serialization.json.CustomGsonFactory;
import com.google.gson.Gson;

public class NetworkSerialization extends TestingTools {
    final public static Gson GSON = CustomGsonFactory.getInstance();

    public static void main(String... args) throws FileAlreadyExistsException, SecurityException, NullPointerException, IOException{
        boolean makeResultsFalse = false;

        MutableNeuralNetwork deserializedNetwork, originalNetwork;
        String file_path = "TestSaves\\test.snn";

        NetworkSerializer serializer = new NetworkSerializer();
        NetworkDeserializer deserializer = new NetworkDeserializer();

        NetworkLayout layout = new NetworkLayoutBuilder()
                               .withInputLayer(1)
                               .withOutputLayer(1)
                               .addLayers(2, 4, ActivationFunctions.TANH)
                               .build();

        originalNetwork = new MutableNeuralNetworkBuilder(layout).build();
        NeuralNetworkTools.randomizeWeightsAndBiases(originalNetwork);

        if(makeResultsFalse){
            layout = new NetworkLayoutBuilder()
                        .withInputLayer(2)
                        .withOutputLayer(4)
                        .addLayers(4, 2, ActivationFunctions.TANH)
                        .build();
            deserializedNetwork = new MutableNeuralNetworkBuilder(layout).build();
            print(formatColorized("\nNOTICE: ", RED) + "Intentionally making results false");

        } else {
            serializer.save(originalNetwork, file_path, true);
            deserializedNetwork = deserializer.load(file_path, new MutableNeuralNetworkBuilder()).build();
        }

        println("\n\nShould be true: " + formatBoolean(originalNetwork.equals(deserializedNetwork)));

        printSideBySide(
            formatColorized("\nOriginal:", BLUE), NetworkLayout.of(originalNetwork).toString(),
            formatColorized("From file:", BLUE), NetworkLayout.of(deserializedNetwork).toString()
        );
        
        printSideBySide(
            formatColorized("\nOriginal:", BLUE), networkToString(originalNetwork),
            formatColorized("From file:", BLUE), networkToString(deserializedNetwork)
        );

        if(makeResultsFalse)
            println(formatColorized("Intentionally False", RED));

        println();
    }

    
    
}

import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;

import com.google.gson.Gson;
import com.mjsd.simpleneuralnetwork.ActivationFunctions;
import com.mjsd.simpleneuralnetwork.NetworkLayout;
import com.mjsd.simpleneuralnetwork.NetworkLayoutBuilder;
import com.mjsd.simpleneuralnetwork.NeuralNetworkTools;
import com.mjsd.simpleneuralnetwork.Serialization.NetworkDeserializer;
import com.mjsd.simpleneuralnetwork.Serialization.NetworkSerializer;
import com.mjsd.simpleneuralnetwork.gson.CustomGsonFactory;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetworkBuilder;

public class NetworkSerialization extends TestingEnvironment {
    final public static Gson GSON = CustomGsonFactory.getInstance();

    public static void main(String... args) throws FileAlreadyExistsException, SecurityException, NullPointerException, IOException{
        MutableNeuralNetwork deserializedNetwork, originalNetwork;
        String file_path = "test.snn";

        NetworkSerializer serializer = new NetworkSerializer();
        NetworkDeserializer deserializer = new NetworkDeserializer();

        NetworkLayout layout = new NetworkLayoutBuilder()
                               .withInputLayer(1)
                               .withOutputLayer(1)
                               .addLayers(2, 4, ActivationFunctions.TANH)
                               .build();

        originalNetwork = new MutableNeuralNetworkBuilder(layout).build();
        NeuralNetworkTools.randomizeWeightsAndBiases(originalNetwork);

        serializer.save(originalNetwork, file_path, true);

        deserializedNetwork = deserializer.load(file_path, new MutableNeuralNetworkBuilder()).build();

        println("\n\nOriginal:\n" + originalNetwork.getLayout());
        println("\n\nFrom file:\n" + deserializedNetwork.getLayout());

        println("\nShould be true: " + originalNetwork.equals(deserializedNetwork));
    }
    
}

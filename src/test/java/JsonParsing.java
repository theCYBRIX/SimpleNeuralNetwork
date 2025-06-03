import java.time.Instant;
import java.util.HashMap;

import com.github.thecybrix.simpleneuralnetwork.core.ActivationFunctions;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetworkBuilder;
import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayout;
import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayoutBuilder;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkTools;
import com.github.thecybrix.simpleneuralnetwork.serialization.json.CustomGsonFactory;
import com.google.gson.Gson;

public class JsonParsing extends TestingTools {
    final public static Gson GSON = CustomGsonFactory.getInstance();

    public static void main(String... args){
        MutableNeuralNetwork serializedNetwork, originalNetwork;

        HashMap<String, String> metadata = new HashMap<>();
        metadata.put("source", "test");
        metadata.put("date", Instant.now().toString());


        String asJson;

        NetworkLayout layout = new NetworkLayoutBuilder()
                               .withInputLayer(1)
                               .withOutputLayer(1)
                               .addLayers(2, 4, ActivationFunctions.TANH)
                               .build();

        MutableNeuralNetworkBuilder builder = new MutableNeuralNetworkBuilder(layout);
        builder.withMetadata(metadata);

        originalNetwork = builder.build();


        asJson = originalNetwork.toJson(GSON);

        double[] inputValues = new double[originalNetwork.getInputLayer().length];
        for(int i = 0; i < inputValues.length; i++)
            inputValues[i] = Math.random();

        NeuralNetworkTools.randomizeWeightsAndBiases(originalNetwork);
        originalNetwork.setInputs(inputValues);
        originalNetwork.forwardPass();


        serializedNetwork = MutableNeuralNetworkBuilder.fromJson(asJson);

        println("\nNetwork Layout:");
        println(NetworkLayout.of(serializedNetwork));

        println("\nNetwork as JSON:");
        println(asJson);

        println("\nShould be false: " + originalNetwork.equals(serializedNetwork));

        asJson = originalNetwork.toJson(GSON);
        serializedNetwork = MutableNeuralNetworkBuilder.fromJson(asJson);

        println("\nNetwork as JSON (Randomized):");
        println(asJson);

        println("\nShould be true: " + originalNetwork.equals(serializedNetwork));
    }
    
}

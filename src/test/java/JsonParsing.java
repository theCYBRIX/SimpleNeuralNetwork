import com.google.gson.Gson;
import com.mjsd.simpleneuralnetwork.ActivationFunctions;
import com.mjsd.simpleneuralnetwork.NetworkLayout;
import com.mjsd.simpleneuralnetwork.NetworkLayoutBuilder;
import com.mjsd.simpleneuralnetwork.NeuralNetworkTools;
import com.mjsd.simpleneuralnetwork.gson.CustomGsonFactory;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import com.mjsd.simpleneuralnetwork.training.MutableNeuralNetworkBuilder;

public class JsonParsing extends TestingEnvironment {
    final public static Gson GSON = CustomGsonFactory.getInstance();

    public static void main(String... args){
        MutableNeuralNetwork serializedNetwork, originalNetwork;
        String asJson;

        NetworkLayout layout = new NetworkLayoutBuilder()
                               .withInputLayer(1)
                               .withOutputLayer(1)
                               .addLayers(2, 4, ActivationFunctions.TANH)
                               .build();

        originalNetwork = new MutableNeuralNetworkBuilder(layout).build();


        asJson = originalNetwork.toJson(GSON);

        double[] inputValues = new double[originalNetwork.getInputLayer().length];
        for(int i = 0; i < inputValues.length; i++)
            inputValues[i] = Math.random();

        NeuralNetworkTools.randomizeWeightsAndBiases(originalNetwork);
        originalNetwork.setInputs(inputValues);
        originalNetwork.forwardPass();


        serializedNetwork = MutableNeuralNetworkBuilder.fromJson(asJson);

        println("\nNetwork Layout:");
        println(serializedNetwork.getLayout());

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

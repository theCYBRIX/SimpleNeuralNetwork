package mjsd.simpleneuralnetwork.gson;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import mjsd.simpleneuralnetwork.NetworkLayout;
import mjsd.simpleneuralnetwork.SimpleNeuralNetwork;
import mjsd.simpleneuralnetwork.NetworkLayout.NetworkLayer;
import mjsd.simpleneuralnetwork.SimpleNeuralNetwork.ActivationFunction;
import mjsd.simpleneuralnetwork.SimpleNeuralNetwork.InputNormalizer;
import mjsd.simpleneuralnetwork.SimpleNeuralNetwork.InputProvider;
import mjsd.simpleneuralnetwork.SimpleNeuralNetwork.OutputHandler;
import mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import mjsd.simpleneuralnetwork.training.RankedNeuralNetwork;

public abstract class CustomGsonFactory {
    private static Gson customGson = null; 

    public static Gson getInstance(){
        if(customGson == null){
            GsonBuilder gsonBuilder = new GsonBuilder();
            gsonBuilder.registerTypeAdapter(ActivationFunction.class, new ActivationFunctionAdapter())
                       .registerTypeAdapter(InputNormalizer.class, new InputNormalizerAdapter())
                       .registerTypeAdapter(InputProvider.class, new InputProviderAdapter())
                       .registerTypeAdapter(OutputHandler.class, new OutputHandlerAdapter())
                       .registerTypeAdapter(NetworkLayer.class, new NetworkLayerAdapter())
                       .registerTypeAdapter(NetworkLayer[].class, new ArrayAdapter<>(NetworkLayer.class, NetworkLayer[]::new))
                       .registerTypeAdapter(NetworkLayout.class, new NetworkLayoutAdapter())
                       .registerTypeAdapter(SimpleNeuralNetwork.class, new SimpleNeuralNetworkAdapter())
                       .registerTypeAdapter(MutableNeuralNetwork.class, new MutableNeuralNetworkAdapter())
                       .registerTypeAdapter(RankedNeuralNetwork.class, new RankedNeuralNetworkAdapter());
            customGson = gsonBuilder.create();
        }

        return customGson;
    }
}

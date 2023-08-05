package mjsd.simpleneuralnetwork.gson;

import mjsd.simpleneuralnetwork.SimpleNeuralNetwork;
import mjsd.simpleneuralnetwork.SimpleNeuralNetworkBuilder;

public class SimpleNeuralNetworkAdapter extends NeuralNetworkAdapter<SimpleNeuralNetwork> {
    public SimpleNeuralNetworkAdapter(){ super(new SimpleNeuralNetworkBuilder()); }
    public SimpleNeuralNetworkAdapter(SimpleNeuralNetworkBuilder builder){ super(builder); }
}

package mjsd.simpleneuralnetwork.gson;

import mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import mjsd.simpleneuralnetwork.training.MutableNeuralNetworkBuilder;

public class MutableNeuralNetworkAdapter extends NeuralNetworkAdapter<MutableNeuralNetwork> {
    public MutableNeuralNetworkAdapter(){ super(new MutableNeuralNetworkBuilder()); }
    public MutableNeuralNetworkAdapter(MutableNeuralNetworkBuilder builder){ super(builder); }
}
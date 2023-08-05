package mjsd.simpleneuralnetwork.gson;

import mjsd.simpleneuralnetwork.training.RankedNeuralNetwork;
import mjsd.simpleneuralnetwork.training.RankedNeuralNetworkBuilder;

public class RankedNeuralNetworkAdapter extends NeuralNetworkAdapter<RankedNeuralNetwork> {
    public RankedNeuralNetworkAdapter(){ super(new RankedNeuralNetworkBuilder()); }
    public RankedNeuralNetworkAdapter(RankedNeuralNetworkBuilder builder){ super(builder); }
}

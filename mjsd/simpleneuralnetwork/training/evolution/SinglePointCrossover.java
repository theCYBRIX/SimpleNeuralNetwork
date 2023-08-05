package mjsd.simpleneuralnetwork.training.evolution;

import java.util.function.Supplier;

import mjsd.simpleneuralnetwork.NeuralNetworkTools;
import mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import mjsd.simpleneuralnetwork.training.evolution.SimpleOffspringProvider.TwoParentOffspringProvider;

public class SinglePointCrossover<E extends MutableNeuralNetwork> extends TwoParentOffspringProvider<E>{

    public SinglePointCrossover(Supplier<E> networkSupplier) throws NullPointerException {
        super((x, y) -> NeuralNetworkTools.singlePointCrossover(x, y, networkSupplier));
    }
    
}

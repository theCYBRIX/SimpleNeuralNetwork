package mjsd.simpleneuralnetwork.training.evolution;

import java.util.function.Supplier;

import mjsd.simpleneuralnetwork.NeuralNetworkTools;
import mjsd.simpleneuralnetwork.training.MutableNeuralNetwork;
import mjsd.simpleneuralnetwork.training.evolution.SimpleOffspringProvider.SingleParentOffspringProvider;

public class RandomMutation<E extends MutableNeuralNetwork> extends SingleParentOffspringProvider<E> {

    public RandomMutation(double maxWeightDeviation, double maxBiasDeviation, Supplier<E> networkSupplier) throws NullPointerException{
        super(x -> NeuralNetworkTools.randomMutation(NeuralNetworkTools.copy(x, networkSupplier), maxWeightDeviation, maxBiasDeviation));
    }

}

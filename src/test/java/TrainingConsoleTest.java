import java.util.concurrent.Executors;

import com.github.thecybrix.simpleneuralnetwork.api.SimpleNNConsole;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetworkBuilder;

public class TrainingConsoleTest {
    public static void main(String[] args) {
        SimpleNNConsole<MutableNeuralNetwork> trainingConsole = new SimpleNNConsole<>(new MutableNeuralNetworkBuilder(), Executors.newWorkStealingPool());
        trainingConsole.run();
    }
}

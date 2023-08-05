package mjsd.simpleneuralnetwork.training;

import java.util.Collection;

public interface TrainingScenario<E extends MutableNeuralNetwork> extends Runnable {
	public void setParticipants(Collection<E> c);
	public void evaluateParticipants();
}
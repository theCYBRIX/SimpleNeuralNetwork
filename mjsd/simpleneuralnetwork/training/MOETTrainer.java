package mjsd.simpleneuralnetwork.training;

import java.lang.Thread.UncaughtExceptionHandler;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Supplier;

import mjsd.simpleneuralnetwork.NeuralNetworkTools;
import mjsd.simpleneuralnetwork.training.evolution.Ecosystem;
import mjsd.simpleneuralnetwork.training.evolution.Population;

//Multi-objective Evolutionary Trainer (MOET)
public class MOETTrainer<E extends RankedNeuralNetwork> implements Runnable{
    final private static float DEF_POPULATION_MIX_FRACTION = 0.1f;

    final private Supplier<E> NETWORK_SUPPLIER; 
    private ArrayList<EvolutionaryTrainer<E>> networkTrainers;
    private int exchangePopulationSize;
    private int numExchangeChildren;
    private TrainingSynchronizer synchronizer = new TrainingSynchronizer();

    private Thread runningThread;
    private ThreadGroup trainingThreads;

    private boolean running;
    private volatile boolean keepTraining;

    private ArrayList<E[]> bestPerformingNetworks;

    private int generation = 0;

    private Object parentWaitingObject = new Object();


    /**
     * @param networkLayout The layout to use for the creation of networks.
     * @param networksPerGeneration The total number of networks to be trained simultaneously across all scenarios.
     * @param scenarios The scenarios in which to train the networks.
     * @throws TooFewParticipantsException
     * @throws NullPointerException
     */
    public MOETTrainer(int networksPerGeneration, List<? extends EvolutionaryTrainer<E>> scenarios, Supplier<E> networkSupplier) throws IllegalArgumentException, NullPointerException{
        this(networksPerGeneration, DEF_POPULATION_MIX_FRACTION, scenarios, null, networkSupplier);
    }

    /**
     * @param networkLayout The layout to use for the creation of networks.
     * @param networksPerGeneration The total number of networks to be trained simultaneously across all scenarios.
     * @param populationMixFraction The fraction of the population in each scenario, that gets used for crossing and interchanging between scenarios. Restricted to range {@code [0, 1]}.
     * @param scenarios The scenarios in which to train the networks.
     * @throws TooFewParticipantsException
     * @throws NullPointerException
     */
    public MOETTrainer(int networksPerGeneration, float populationMixFraction, List<? extends EvolutionaryTrainer<E>> scenarios, Supplier<E> networkSupplier) throws IllegalArgumentException, NullPointerException{
        this(networksPerGeneration, populationMixFraction, scenarios, null, networkSupplier);
    }


    /**
     * @param networkLayout The layout to use for the creation of networks.
     * @param networksPerGeneration The total number of networks to be trained simultaneously across all scenarios.
     * @param scenarios The scenarios in which to train the networks.
     * @param postGenCallbacks Callbacks to be called after each scenario completes a generation.
     * @throws TooFewParticipantsException
     * @throws NullPointerException
     */
    public MOETTrainer(int networksPerGeneration, List<? extends EvolutionaryTrainer<E>> scenarios, List<Consumer<EvolutionaryTrainer<E>>> postGenCallbacks, Supplier<E> networkSupplier) throws IllegalArgumentException, NullPointerException{
        this(networksPerGeneration, DEF_POPULATION_MIX_FRACTION, scenarios, postGenCallbacks, networkSupplier);
    }

    /**
     * @param networkLayout The layout to use for the creation of networks.
     * @param networksPerGeneration The total number of networks to be trained simultaneously across all scenarios.
     * @param populationMixFraction The fraction of the population in each scenario, that gets used for crossing and interchanging between scenarios. Restricted to range {@code [0, 1]}.
     * @param scenarios The scenarios in which to train the networks.
     * @param postGenCallbacks Callbacks to be called after each scenario completes a generation.
     * @throws TooFewParticipantsException
     * @throws NullPointerException
     */
    public MOETTrainer(int networksPerGeneration, float populationMixFraction, List<? extends EvolutionaryTrainer<E>> scenarios, List<? extends Consumer<EvolutionaryTrainer<E>>> postGenCallbacks, Supplier<E> networkSupplier) throws IllegalArgumentException, NullPointerException{
        if(populationMixFraction < 0 || populationMixFraction > 1) throw new IllegalArgumentException("Population mix fraction must be within the range [0, 1].");
        if(networksPerGeneration < 0) throw new IllegalArgumentException("Trainer cannot have less than 1 network per generation.");
        networkTrainers = new ArrayList<EvolutionaryTrainer<E>>(Objects.requireNonNull(scenarios).size());

        if(postGenCallbacks != null && postGenCallbacks.size() != scenarios.size())
            throw new IllegalArgumentException("The trainer must have either no PostGenerationCallbacks (null), or the same number of callbacks as scenarios.");

        NETWORK_SUPPLIER = Objects.requireNonNull(networkSupplier, "NetworkSupplier is null.");

        exchangePopulationSize = Math.round(networksPerGeneration * populationMixFraction);

        numExchangeChildren = Math.max(exchangePopulationSize - (scenarios.size() * (scenarios.size() - 1)), 0);

        int networksPerEcosystem = Math.round((networksPerGeneration / (float)scenarios.size()) - exchangePopulationSize);
        
        bestPerformingNetworks = new ArrayList<>(exchangePopulationSize);

        EvolutionaryTrainer<E> trainer;
        for(int i = 0; i < scenarios.size(); i++){
            trainer = scenarios.get(i);
            trainer.attachCallback(synchronizer);
            networkTrainers.add(trainer);
        }

        if(postGenCallbacks != null) 
            for(int i = 0; i< networkTrainers.size(); i++)
                networkTrainers.get(i).attachCallback(postGenCallbacks.get(i));
    }

    @Override
    public void run() {

        if(running) return;

        running = true;

        runningThread = Thread.currentThread();
        String previousThreadName = runningThread.getName();

        try {

            runningThread.setName("Training manager");
            runningThread.setUncaughtExceptionHandler(synchronizer);
        
            keepTraining = true;
            trainingThreads = new ThreadGroup("Training Threads");

            Thread trainingThread;
            for(int i = 0; i < networkTrainers.size(); i++){
                trainingThread = new Thread(trainingThreads, networkTrainers.get(i));
                trainingThread.setUncaughtExceptionHandler(synchronizer);
                trainingThread.start();
            }


            while(keepTraining){

                try {
                    synchronized(parentWaitingObject){
                        parentWaitingObject.wait();
                    }
                } catch (InterruptedException e) {
                    break;
                }

    
                synchronized(bestPerformingNetworks){
                    bestPerformingNetworks.clear();

                    if(exchangePopulationSize > 0){
                        ArrayList<E> bestOfEachPopulation = new ArrayList<>(networkTrainers.size());

                        //Exchange children of best performing networks between scenarios
                        /*
                        for(EvolutionaryTrainer<E> scenario : networkTrainers){
                            Ecosystem<E> population = scenario.getEcosystem();
                            E[] bestOfPopulation = population.getEliteNetworks();
                            bestPerformingNetworks.add(bestOfPopulation);

                            for(EvolutionaryTrainer<E> otherScenario : networkTrainers){
                                if(otherScenario == scenario) continue;
                                otherScenario.add(bestOfPopulation[0]);
                            }

                            bestOfEachPopulation.add(bestOfPopulation[0]);
                        } */

                        if(numExchangeChildren > 0 && networkTrainers.size() > 1){
                            LinkedList<E> crossovers = new LinkedList<>();

                            int index1 = -1, index2 = bestOfEachPopulation.size();
                            while(crossovers.size() < numExchangeChildren){
                                index1++;
                                index2--;
                                if(index1 >= bestOfEachPopulation.size()) index1 = 0;
                                if(index2 < 0) index2 = bestOfEachPopulation.size() - 1;
                                if(index1 == index2){
                                    index2++;
                                    continue;
                                }
                                
                                E parent1 = bestOfEachPopulation.get(index1),
                                            parent2 = bestOfEachPopulation.get(index2);
                                crossovers.add(NeuralNetworkTools.singlePointCrossover(parent1, parent2, NETWORK_SUPPLIER));
                            }

                            while(crossovers.size() > numExchangeChildren){
                                crossovers.removeLast();
                            }

                            for(EvolutionaryTrainer<E> scenario : networkTrainers){
                                scenario.addAll(crossovers);
                            }
                        }

                    }


                }

                generation++;

                try {
                    synchronized(trainingThreads){
                        trainingThreads.notifyAll();
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                    break;
                }
            }

        } finally {
            keepTraining = false;
            running = false;

            if(trainingThreads != null) trainingThreads.interrupt();

            runningThread.setName(previousThreadName);
            runningThread = null;
        }
    }

    public void stop() throws SecurityException {
        keepTraining = false;
        if(runningThread != null) runningThread.interrupt();
    }

    public boolean isRunning() {
        return running;
    }

    public Optional<Thread> getRunningThread() {
        return Optional.ofNullable(runningThread);
    }

    public int getGeneration() {
        return generation;
    }

    public void setGeneration(int generation) {
        this.generation = generation;
    }

    /*
    public List<Population<E>> getPopulations(){
        ArrayList<Population<E>> populations = new ArrayList<>(networkTrainers.size());

        for(Population<E> p : networkTrainers){
            populations.add(p.getEcosystem());
        }
        
        return current;
    } */

    public List<EvolutionaryTrainer<E>> getTrainers(){
        return new LinkedList<>(networkTrainers);
    }

    public void addNetwork(E network) throws NullPointerException {
        for(EvolutionaryTrainer<E> ecosystem : networkTrainers)
            ecosystem.add(network);
    }

    public ArrayList<EvolutionaryTrainer<E>> getNetworkTrainers() {
        return networkTrainers;
    }


    

    private class TrainingSynchronizer implements Consumer<EvolutionaryTrainer<E>>, UncaughtExceptionHandler{
        private volatile int waitingCount = 0;

        @Override
        public void accept(EvolutionaryTrainer<E> trainer) {
            if(!networkTrainers.contains(trainer)) return;

            try {
                waitingCount++;

                //Wake the parent thread (Thread running the MOETTrainer)
                if(waitingCount == networkTrainers.size()){
                    synchronized(parentWaitingObject){
                        parentWaitingObject.notifyAll();
                    }
                }
            
                if(runningThread.isAlive()){
                    synchronized(trainingThreads){
                        trainingThreads.wait();
                    }
                } else{
                    throw new InterruptedException();
                }

            } catch (InterruptedException e) {
                trainer.stop();
            } finally {
                waitingCount--;
            }
        }

        @Override
        public void uncaughtException(Thread t, Throwable e) {
            e.printStackTrace();

            trainingThreads.interrupt();
            trainingThreads.getParent().interrupt();
        }
    }
}

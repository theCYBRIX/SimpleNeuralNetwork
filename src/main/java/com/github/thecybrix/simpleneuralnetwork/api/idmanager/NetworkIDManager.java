package com.github.thecybrix.simpleneuralnetwork.api.idmanager;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayout;
import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.exceptions.DimensionsMismatchException;
import com.github.thecybrix.simpleneuralnetwork.server.APIContext;
import com.github.thecybrix.simpleneuralnetwork.server.BinaryIOHandler;
import com.github.thecybrix.simpleneuralnetwork.server.JsonRequestHandler;
import com.github.thecybrix.simpleneuralnetwork.server.SimpleTCPServer;
import com.github.thecybrix.simpleneuralnetwork.util.IDManager;

import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;

public class NetworkIDManager<E extends SimpleNeuralNetwork> implements APIContext {
    final private static AtomicInteger INSTANCE_COUNTER = new AtomicInteger();
    final public static int DEFAULT_IO_PORT = 3075;
    
    final private Logger logger;
    final public ForkJoinPool forkJoinPool = new ForkJoinPool();

    protected Map<Integer, E> neuralNetworks = new HashMap<>();
    protected IDManager idManager = new IDManager();
    
    private SimpleTCPServer binaryChannel = null;

    public NetworkIDManager(){
        this(Integer.toString(INSTANCE_COUNTER.incrementAndGet()));
    }

    public NetworkIDManager(String instanceID){
        logger = Logger.getLogger(NetworkIDManager.class.getName() + "-" + Objects.requireNonNull(instanceID, "Instance ID is null."));
    }

    public Logger getLogger() {
        return logger;
    }
    
    public NetworkLayout getLayout(int id) throws NoSuchElementException {
        return NetworkLayout.of(getNetwork(id));
    }
    
    public E getNetwork(int id) throws NoSuchElementException {
        E network = neuralNetworks.get(id);
        if (network == null)
            throw new NoSuchElementException(getInvalidIdString(id));
        return network;
    }

    public int[] addAll(List<? extends E> networks) throws NullPointerException{
        Objects.requireNonNull(networks, "Networks list is null.");
        for(E n : networks)
            if (n == null) throw new NullPointerException("Networks list contains null.");
        
        int[] ids = new int[networks.size()];
        int index = 0;
        for (E n : networks)
            ids[index++] = add(n);
        
        return ids;
    }

    public int add(E network){
        int id = idManager.getNextID();
        neuralNetworks.put(id, network);
        return id;
    }

    public E remove(Integer id) throws NoSuchElementException, NullPointerException{
        E network = neuralNetworks.remove(id);
        if (network == null)
            throw new NoSuchElementException(getInvalidIdString(id));
        return network;
    }

    public boolean removeAll(Collection<Integer> ids) throws NoSuchElementException, IllegalArgumentException, NullPointerException{
        validateIdCollection(ids);
        for (Integer i : ids){
            if(neuralNetworks.remove(i) != null)
                idManager.releaseID(i);
        }
        return true;
    }

    public SimpleTCPServer openBinaryChannel(){
        return openBinaryChannel(DEFAULT_IO_PORT);
    }

    public SimpleTCPServer openBinaryChannel(int port){
        closeBinaryChannel();
        BinaryIOHandler ioHandler = new BinaryIOHandler();
        ioHandler.addRequestHandler(new BinaryProcessInputsRequest<>(this));
        binaryChannel = new SimpleTCPServer(port, ioHandler);
        binaryChannel.start(BinaryIOHandler.class.getSimpleName());
        return binaryChannel;
    }

    public boolean closeBinaryChannel(){
        if(binaryChannel == null || !binaryChannel.isRunning())
            return false;
        binaryChannel.stop();
        return true;
    }

    public Int2ObjectMap<double[]> processInputs(Int2ObjectMap<double[]> inputData){
        Int2ObjectMap<double[]> outputs = new Int2ObjectOpenHashMap<double[]>(inputData.size());
        ArrayList<Integer> failedIds = new ArrayList<>();

        forkJoinPool.submit(
            () -> inputData.int2ObjectEntrySet().parallelStream().forEach(
                (entry) -> {
                    int id = entry.getIntKey();
                    try {
                        double[] result = process(id, entry.getValue());
                        synchronized(outputs){
                            outputs.put(id, result);
                        }
                        
                    } catch (DimensionsMismatchException | NullPointerException e) {
                        synchronized(failedIds){
                            failedIds.add(id);
                        }
                    }
                }
            )
        ).join();
        
        if(!failedIds.isEmpty())
            logger.warning("Failed to process forward passes: " + failedIds);

        return outputs;
    }

    private double[] process(int networkIndex, double[] inputs) throws DimensionsMismatchException, NullPointerException {
        E network = neuralNetworks.get(networkIndex);
        synchronized(network){
            network.setInputs(inputs);
            network.forwardPass();
            return network.getOutputs();
        }
    }

    public HashMap<Integer, Map<String, Object>> getMetadata(List<Integer> ids){
        HashMap<Integer, Map<String, Object>> metadataPacket = new HashMap<>(ids.size());
        ids.stream().forEach(x -> metadataPacket.put(x, neuralNetworks.get(x).getMetadata()));
        return metadataPacket;
    }

    public HashMap<Integer, Map<String, Object>> getMetadata(){
        return getMetadata(neuralNetworks.keySet().stream().collect(Collectors.toList()));
    }

    public Map<Integer, E> getNetworks(){
        HashMap<Integer, E> networks = new HashMap<>(neuralNetworks.size());
        neuralNetworks.entrySet().forEach(x -> networks.put(x.getKey(), x.getValue()));
        return networks;
    }

    public Map<Integer, E> getNetworks(List<Integer> ids) throws IllegalArgumentException, NullPointerException {
        Objects.requireNonNull(ids, "ID list is null.");
        if(ids.isEmpty()) throw new IllegalArgumentException("ID list is empty.");

        Integer[] invalidIds = ids.stream().filter(x -> !neuralNetworks.containsKey(x)).toArray(Integer[]::new);
        if(invalidIds.length != 0)
            throw new IllegalArgumentException(getInvalidIdString(invalidIds));
        HashMap<Integer, E> networks = new HashMap<>(ids.size());
        ids.stream().forEach(x -> networks.put(x, neuralNetworks.get(x)));
        return networks; 
    }

    public List<JsonRequestHandler> getRequestHandlers(){
        return Arrays.asList(
            new ProcessInputsRequest<>(this),
            new AddNetworkRequest<>(this),
            new RemoveNetworkRequest<>(this),
            new GetNetworkRequest<>(this),
            new GetMetadataRequest<>(this),
            new OpenBinaryChannelRequest<>(this),
            new CloseBinaryChannelRequest<>(this)
        );
    }

    private void validateIdCollection(Collection<Integer> ids) throws NoSuchElementException, IllegalArgumentException, NullPointerException{
        validateIdCollection(ids, neuralNetworks);
    }

    public static void validateIdCollection(Collection<Integer> ids, Map<Integer, ?> sourceMap) throws NoSuchElementException, IllegalArgumentException, NullPointerException{
        Objects.requireNonNull(ids, "ID collection is null.");
        if(ids.isEmpty()) throw new IllegalArgumentException("ID collection is empty.");
        if(ids.stream().anyMatch(x -> x == null)) throw new NullPointerException("ID collection contains null.");
        if(!sourceMap.keySet().containsAll(ids)) throw new NoSuchElementException(getInvalidIdString(ids, sourceMap));
    }

    public static String getInvalidIdString(Collection<Integer> ids, Map<Integer, ?> sourceMap){
        return getInvalidIdString(ids.stream().filter(x -> !sourceMap.containsKey(x)).toArray(Integer[]::new));
    }

    public static String getInvalidIdString(Integer id) throws NullPointerException {
        Objects.requireNonNull(id, "ID is null.");
        return "No network with ID \"" + id + "\" available.";
    }

    public static String getInvalidIdString(Integer... ids) throws NullPointerException, IllegalArgumentException{
        Objects.requireNonNull(ids, "ID array is null.");
        if(ids.length == 0) throw new IllegalArgumentException("ID array is empty.");
        StringBuilder message = new StringBuilder("Invalid network IDs requested: ");
        message.append("[").append(ids[0]);
        for(int i = 0; i < ids.length; i++)
            message.append(", ").append(ids[i]);
        message.append("]");

        return message.toString();
    }
}

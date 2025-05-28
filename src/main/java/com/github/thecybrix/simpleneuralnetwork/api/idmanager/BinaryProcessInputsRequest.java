package com.github.thecybrix.simpleneuralnetwork.api.idmanager;

import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.server.ContextualBinaryRequestHandler;
import com.github.thecybrix.simpleneuralnetwork.util.EndianAwareInputStream;
import com.github.thecybrix.simpleneuralnetwork.util.EndianAwareOutputStream;

import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;

public class BinaryProcessInputsRequest<E extends SimpleNeuralNetwork> extends ContextualBinaryRequestHandler<NetworkIDManager<E>> {
    final private static byte DEFAULT_ENDPOINT = 2;
    

    public BinaryProcessInputsRequest(NetworkIDManager<E> context) throws NullPointerException {
        super(context, DEFAULT_ENDPOINT);
    }
    
    
    public BinaryProcessInputsRequest(NetworkIDManager<E> context, int endpoint) throws NullPointerException {
        super(context, endpoint);
    }

    /**
     * <p> input: numNetworks (int) -> [loopNetworks] -> networkId (int) -> numDoubles(int) -> [loopDoubles] -> nodeValue (double) -> [/loopDoubles] ->  [/loopNetworks] </p>
     * <p> response: numNetworks (int) -> [loopNetworks] -> networkId (int) -> numDoubles(int) -> [loopDoubles] -> nodeValue (double) -> [/loopDoubles] ->  [/loopNetworks] </p>
    **/
    @Override
    public void handle(EndianAwareInputStream input, EndianAwareOutputStream output, NetworkIDManager<E> context) throws Exception {
        
        // StopWatch stopWatch = new StopWatch();
        // stopWatch.start();

        Int2ObjectMap<double[]> inputMap = new Int2ObjectOpenHashMap<>();

        int numNetworks = input.readInt();
        for (int network = 0; network < numNetworks; network++) {
            int networkId = input.readInt();
            int numInputs = input.readInt();
            double[] nodeValues = input.readDoubleArray(numInputs);

            inputMap.put(networkId, nodeValues);
        }

        Int2ObjectMap<double[]> outputMap = context.processInputs(inputMap);

        output.writeInt(0); // OK signal

        output.writeInt(outputMap.size());

        for (Int2ObjectMap.Entry<double[]> entry : outputMap.int2ObjectEntrySet()) {
            int networkId = entry.getIntKey();
            double[] nodeValues = entry.getValue();

            output.writeInt(networkId);
            output.writeInt(nodeValues.length);

            output.writeDoubleArray(nodeValues);
        }

        output.flush();

        // stopWatch.stop();
        // DecimalFormat format = new DecimalFormat("#.####");
        // System.out.println("ms: " + format.format(stopWatch.getMillisExact()));
    }
    
}

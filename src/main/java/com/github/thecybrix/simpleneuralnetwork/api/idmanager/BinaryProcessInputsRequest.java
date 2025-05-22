package com.github.thecybrix.simpleneuralnetwork.api.idmanager;

import java.io.InputStream;
import java.io.OutputStream;

import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.server.ContextualBinaryRequestHandler;
import com.github.thecybrix.simpleneuralnetwork.util.EndianConverter;

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
    public void handle(InputStream input, OutputStream output, boolean bigEndian, NetworkIDManager<E> context) throws Exception {
        Int2ObjectMap<double[]> inputMap = new Int2ObjectOpenHashMap<>();

        int numNetworks = EndianConverter.bytesToInt(input.readNBytes(4), bigEndian);
        for (int network = 0; network < numNetworks; network++) {
            int networkId = EndianConverter.bytesToInt(input.readNBytes(4), bigEndian);
            int numInputs = EndianConverter.bytesToInt(input.readNBytes(4), bigEndian);
            double[] nodeValues = new double[numInputs];
            for(int nodeIndex = 0; nodeIndex < numInputs; nodeIndex++){
                nodeValues[nodeIndex] = EndianConverter.bytesToDouble(input.readNBytes(8), bigEndian);
            }
            inputMap.put(networkId, nodeValues);
        }

        Int2ObjectMap<double[]> outputMap = context.processInputs(inputMap);

        output.write(0); //OK Signal

        byte[] numNetworksBytes = EndianConverter.intToBytes(outputMap.size(), bigEndian);
        output.write(numNetworksBytes);

        for (Int2ObjectMap.Entry<double[]> entry : outputMap.int2ObjectEntrySet()) {
            int networkId = entry.getIntKey();
            double[] nodeValues = entry.getValue();
            
            output.write(EndianConverter.intToBytes(networkId, bigEndian));
            output.write(EndianConverter.intToBytes(nodeValues.length, bigEndian));

            for(double value : nodeValues)
                output.write(EndianConverter.doubleToBytes(value, bigEndian));
        }
    }
    
}

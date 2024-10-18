package com.github.thecybrix.simpleneuralnetwork.serialization.json;

import java.io.IOException;

import com.github.thecybrix.simpleneuralnetwork.core.ActivationFunction;
import com.github.thecybrix.simpleneuralnetwork.core.InputNormalizer;
import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayout.NetworkLayer;
import com.google.gson.Gson;
import com.google.gson.JsonParseException;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

public class NetworkLayerAdapter extends TypeAdapter<NetworkLayer> {
    final private static String NODES = "nodes", ACTIVATION_FUNCTION = "activationFunction", INPUT_NORMALIZER = "inputNormalizer";

    @Override
    public void write(JsonWriter out, NetworkLayer value) throws IOException {
        if(value == null){
            out.nullValue();
            return;
        }

        final Gson GSON = CustomGsonFactory.getInstance();

        out.beginObject();
        out.name(NODES).value(value.getNodeCount());
        out.name(ACTIVATION_FUNCTION).jsonValue(GSON.toJson(value.getActivationFunction(), ActivationFunction.class));
        out.name(INPUT_NORMALIZER).jsonValue(GSON.toJson(value.getInputNormalizer(), InputNormalizer.class));
        out.endObject();
    }

    @Override
    public NetworkLayer read(JsonReader in) throws IOException {
        int numNodes = 0;
        ActivationFunction activationFunction = null;
        InputNormalizer inputNormalizer = null;

        final Gson GSON = CustomGsonFactory.getInstance();

        try {
            in.beginObject();
            while(in.hasNext()){
                switch(in.peek()){
                    case NAME:
                        String var = in.nextName();
                        switch(var){
                            case NODES:
                                numNodes = in.nextInt();
                                break;
                            case ACTIVATION_FUNCTION:
                                activationFunction = GSON.fromJson(in, ActivationFunction.class);
                                break;
                            case INPUT_NORMALIZER:
                                inputNormalizer = GSON.fromJson(in, InputNormalizer.class);
                                break;
                            default:
                                if(!in.isLenient()) throw new JsonParseException("Parser encountered an unexpected field. (" + var + ")");
                                in.skipValue();
                                break;
                        }
                        break;
                    default:
                        if(!in.isLenient()) throw new JsonParseException("Parser encountered an unexpected token. (" + in.peek() + ")");
                        in.skipValue();
                        break;
                        
                }
            }
            in.endObject();

            if(numNodes <= 0 || activationFunction == null || inputNormalizer == null)
                throw new JsonParseException("Parser is unable to populate all necessary fields with given string.");
            
            return new NetworkLayer(numNodes, inputNormalizer, activationFunction);

        } catch (Exception e) {
            if(in.isLenient()) return null;
            throw new JsonParseException("Unable to create " + NetworkLayer.class + " from the given string.", e);
        }
    }
    
}

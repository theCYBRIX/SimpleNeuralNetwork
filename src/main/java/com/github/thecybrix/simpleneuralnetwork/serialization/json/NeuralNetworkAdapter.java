package com.github.thecybrix.simpleneuralnetwork.serialization.json;

import java.io.IOException;
import java.util.HashMap;
import java.util.Objects;

import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayout;
import com.github.thecybrix.simpleneuralnetwork.core.NeuralNetworkBuilder;
import com.github.thecybrix.simpleneuralnetwork.core.SimpleNeuralNetwork;
import com.google.gson.Gson;
import com.google.gson.JsonParseException;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

public class NeuralNetworkAdapter<E extends SimpleNeuralNetwork> extends TypeAdapter<E> {
    final private static String LAYOUT = "layout", WEIGHTS = "weights", BIASES = "biases", METADADA = "metadata";
    final private NeuralNetworkBuilder<E> NETWORK_BUILDER;

    public NeuralNetworkAdapter(NeuralNetworkBuilder<E> networkBuilder){
        this.NETWORK_BUILDER = Objects.requireNonNull(networkBuilder);
    }

    @Override
    public void write(JsonWriter out, E value) throws IOException {
        if(value == null){
            out.nullValue();
            return;
        }

        final Gson GSON = CustomGsonFactory.getInstance();
        
        out.beginObject();
        out.name(LAYOUT).jsonValue(GSON.toJson(NetworkLayout.of(value), NetworkLayout.class));
        out.name(WEIGHTS).jsonValue(GSON.toJson(value.getWeights(), double[][][].class));
        out.name(BIASES).jsonValue(GSON.toJson(value.getBiases(), double[][].class));
        if(out.getSerializeNulls() || !value.getMetadata().isEmpty())
            out.name(METADADA).jsonValue(GSON.toJson(value.getMetadata(), HashMap.class));
        out.endObject();

    }

    @Override
    public E read(JsonReader in) throws IOException {
        final Gson GSON = CustomGsonFactory.getInstance();

        double[][][] weights = null;
        double[][] biases = null;
        NetworkLayout layout = null;
        HashMap<String, String> metadata = null;

        try {
            in.beginObject();
            while(in.hasNext()){
                switch(in.peek()){
                    case NAME:
                        String var = in.nextName();
                        switch(var){
                            case LAYOUT:
                                layout = GSON.fromJson(in, NetworkLayout.class);
                                break;
                            case WEIGHTS:
                                weights = GSON.fromJson(in, double[][][].class);
                                break;
                            case BIASES:
                                biases = GSON.fromJson(in, double[][].class);
                                break;
                            case METADADA:
                                metadata = GSON.fromJson(in, HashMap.class);
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
        
            if(layout == null || weights == null || biases == null)
                throw new JsonParseException("Parser unable to find all necessary attributes in string.");

            NETWORK_BUILDER.reset()
                            .withLayout(layout)
                            .withWeights(weights)
                            .withBiases(biases)
                            .withMetadata(metadata);
            
            
            return NETWORK_BUILDER.build();

        } catch (Exception e) {
            if(in.isLenient()) return null;
            throw new JsonParseException("Unable to construct " + SimpleNeuralNetwork.class + "from String", e);
        }
    }
    
}

package com.mjsd.simpleneuralnetwork.gson;

import java.io.IOException;

import com.google.gson.JsonParseException;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.InputProvider;

public class InputProviderAdapter extends TypeAdapter<InputProvider>{

    @Override
    public void write(JsonWriter out, InputProvider value) throws IOException {
        out.value(value.toString());
    }

    @Override
    public InputProvider read(JsonReader in) throws IOException {
        String providerName = in.nextString();
        try {
            if(providerName.equals(InputProvider.NO_PROVIDER_STRING)) return InputProvider.NO_PROVIDER;
            throw new JsonParseException("Unknown InputProvider. (" + providerName + ")");
        } catch (Exception e) {
            if(in.isLenient()) return (providerName.equalsIgnoreCase(InputProvider.NO_PROVIDER_STRING)) ? InputProvider.NO_PROVIDER : null;
            throw new JsonParseException(e);
        }
    }

}

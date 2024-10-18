package com.github.thecybrix.simpleneuralnetwork.serialization.json;

import java.io.IOException;

import com.github.thecybrix.simpleneuralnetwork.core.InputNormalizer;
import com.github.thecybrix.simpleneuralnetwork.core.InputNormalizers;
import com.google.gson.JsonParseException;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

public class InputNormalizerAdapter extends TypeAdapter<InputNormalizer>{

    @Override
    public void write(JsonWriter out, InputNormalizer value) throws IOException {
        out.value(value.toString());
    }

    @Override
    public InputNormalizer read(JsonReader in) throws IOException {
        String normalizerName = in.nextString();

        try {
            return InputNormalizers.valueOf(normalizerName);
        } catch (Exception e) {
            if(in.isLenient()) return null;
            throw new JsonParseException(e);
        }
    }

}

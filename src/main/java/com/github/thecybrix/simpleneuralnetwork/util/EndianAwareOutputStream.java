package com.github.thecybrix.simpleneuralnetwork.util;

import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Objects;

public class EndianAwareOutputStream implements Closeable {

    private final OutputStream out;
    private final EndianConverter converter;

    public EndianAwareOutputStream(OutputStream outputStream, Endianness endian) {
        this.out = Objects.requireNonNull(outputStream, "Stream is null.");
        this.converter = new EndianConverter(endian);
    }
    
    /**
     * @deprecated Use {@link #writeByte(byte)} instead for clearer naming.
     */
    @Deprecated
    public void write(int b) throws IOException {
        writeByte((byte) b);
    }

    public void writeByte(byte b) throws IOException {
        out.write(b);
    }

    public void writeNBytes(byte[] bytes) throws IOException {
        out.write(bytes);
    }

    public void writeInt(int value) throws IOException {
        out.write(converter.intToBytes(value));
    }

    public void writeDouble(double value) throws IOException {
        out.write(converter.doubleToBytes(value));
    }

    public void writeFloat(float value) throws IOException {
        out.write(converter.floatToBytes(value));
    }

    public void writeLong(long value) throws IOException {
        out.write(converter.longToBytes(value));
    }

    public void writeShort(short value) throws IOException {
        out.write(converter.shortToBytes(value));
    }

    public void writeChar(char value) throws IOException {
        out.write(converter.charToBytes(value));
    }

    public void writeCharArray(char[] array) throws IOException, NullPointerException {
        byte[] bytes = new byte[array.length * 2];
        for (int i = 0; i < array.length; i++) {
            converter.charToBytes(array[i], bytes, i * 2);
        }
        out.write(bytes);
    }

    public void writeShortArray(short[] array) throws IOException, NullPointerException {
        byte[] bytes = new byte[array.length * 2];
        for (int i = 0; i < array.length; i++) {
            converter.shortToBytes(array[i], bytes, i * 2);
        }
        out.write(bytes);
    }

    public void writeIntArray(int[] array) throws IOException, NullPointerException {
        byte[] bytes = new byte[array.length * 4];
        for (int i = 0; i < array.length; i++) {
            converter.intToBytes(array[i], bytes, i * 4);
        }
        out.write(bytes);
    }
    
    public void writeLongArray(long[] array) throws IOException, NullPointerException {
        byte[] bytes = new byte[array.length * 8];
        for (int i = 0; i < array.length; i++) {
            converter.longToBytes(array[i], bytes, i * 8);
        }
        out.write(bytes);
    }

    public void writeFloatArray(float[] array) throws IOException, NullPointerException {
        byte[] bytes = new byte[array.length * 4];
        for (int i = 0; i < array.length; i++) {
            converter.floatToBytes(array[i], bytes, i * 4);
        }
        out.write(bytes);
    }

    public void writeDoubleArray(double[] array) throws IOException, NullPointerException {
        byte[] bytes = new byte[array.length * 8];
        for (int i = 0; i < array.length; i++) {
            converter.doubleToBytes(array[i], bytes, i * 8);
        }
        out.write(bytes);
    }


    public void flush() throws IOException {
        out.flush();
    }
    
    @Override
    public void close() throws IOException {
        out.close();
    }
}


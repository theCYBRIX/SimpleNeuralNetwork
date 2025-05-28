package com.github.thecybrix.simpleneuralnetwork.util;

import java.io.Closeable;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;

public class EndianAwareInputStream implements Closeable {
    
    private final InputStream in;
    private final EndianConverter converter;


    public EndianAwareInputStream(InputStream inputStream, Endianness endian) {
        this.in = Objects.requireNonNull(inputStream, "Stream is null.");
        this.converter = new EndianConverter(endian);
    }
    
    /**
     * @deprecated Use {@link #readByte()} instead for clearer naming.
     */
    @Deprecated
    public int read() throws IOException {
        return readByte();
    }

    public byte readByte() throws IOException {
        return (byte) readUnsignedByte();
    }

    public byte[] readNBytes(int length) throws IOException {
        return in.readNBytes(length);
    }
    
    public int readInt() throws IOException {
        return converter.bytesToInt(in.readNBytes(4));
    }

    public double readDouble() throws IOException {
        return converter.bytesToDouble(in.readNBytes(8));
    }

    public float readFloat() throws IOException {
        return converter.bytesToFloat(in.readNBytes(4));
    }

    public long readLong() throws IOException {
        return converter.bytesToLong(in.readNBytes(8));
    }

    public short readShort() throws IOException {
        return converter.bytesToShort(in.readNBytes(2));
    }

    public char readChar() throws IOException {
        return converter.bytesToChar(in.readNBytes(2));
    }
    
    public int[] readIntArray(int count) throws IOException {
        byte[] buffer = in.readNBytes(count * 4);
        int[] result = new int[count];
        for (int i = 0; i < count; i++) {
            result[i] = converter.bytesToInt(buffer, i * 4);
        }
        return result;
    }

    public double[] readDoubleArray(int count) throws IOException {
        byte[] buffer = in.readNBytes(count * 8);
        double[] result = new double[count];
        for (int i = 0; i < count; i++) {
            result[i] = converter.bytesToDouble(buffer, i * 8);
        }
        return result;
    }

    public float[] readFloatArray(int count) throws IOException {
        byte[] buffer = in.readNBytes(count * 4);
        float[] result = new float[count];
        for (int i = 0; i < count; i++) {
            result[i] = converter.bytesToFloat(buffer, i * 4);
        }
        return result;
    }

    public long[] readLongArray(int count) throws IOException {
        byte[] buffer = in.readNBytes(count * 8);
        long[] result = new long[count];
        for (int i = 0; i < count; i++) {
            result[i] = converter.bytesToLong(buffer, i * 8);
        }
        return result;
    }

    public short[] readShortArray(int count) throws IOException {
        byte[] buffer = in.readNBytes(count * 2);
        short[] result = new short[count];
        for (int i = 0; i < count; i++) {
            result[i] = converter.bytesToShort(buffer, i * 2);
        }
        return result;
    }

    public char[] readCharArray(int count) throws IOException {
        byte[] buffer = in.readNBytes(count * 2);
        char[] result = new char[count];
        for (int i = 0; i < count; i++) {
            result[i] = converter.bytesToChar(buffer, i * 2);
        }
        return result;
    }
    
    @Override
    public void close() throws IOException {
        in.close();
    }
    
    public final int readUnsignedByte() throws IOException {
        int ch = in.read();
        if (ch < 0)
            throw new EOFException();
        return ch;
    }
}

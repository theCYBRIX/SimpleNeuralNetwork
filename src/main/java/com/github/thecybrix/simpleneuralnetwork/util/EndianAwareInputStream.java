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

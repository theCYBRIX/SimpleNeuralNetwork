package com.github.thecybrix.util;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.function.IntFunction;

public class LengthPrefixedWriter extends BufferedOutputStream {
    final private IntFunction<byte[]> INT_TO_BYTES;
    final private boolean BIG_ENDIAN;

    /**
     * Constructs a new LengthPrefixedWriter with the specified output stream.
     * 
     * @param out The output stream to write to.
     */
    public LengthPrefixedWriter(OutputStream out) {
        this(out, true);
    }

    /**
     * Constructs a new LengthPrefixedWriter with the specified output stream
     * and buffer size.
     * 
     * @param out The output stream to write to.
     * @param size The buffer size.
     */
    public LengthPrefixedWriter(OutputStream out, int size) {
        this(out, size, true);
    }

    /**
     * Constructs a new LengthPrefixedWriter with the specified output stream.
     * 
     * @param out The output stream to write to.
     * @param bigEndian Weather to write the length prefix as big-endian.
     */
    public LengthPrefixedWriter(OutputStream out, boolean bigEndian) {
        super(out);
        BIG_ENDIAN = bigEndian;
        INT_TO_BYTES = BIG_ENDIAN ? LengthPrefixedWriter::bigEndianIntToBytes :  LengthPrefixedWriter::littleEndianIntToBytes;
    }

    /**
     * Constructs a new LengthPrefixedWriter with the specified output stream
     * and buffer size.
     * 
     * @param out The output stream to write to.
     * @param size The buffer size.
     * @param bigEndian Weather to write the length prefix as big-endian.
     */
    public LengthPrefixedWriter(OutputStream out, int size, boolean bigEndian) {
        super(out, size);
        BIG_ENDIAN = bigEndian;
        INT_TO_BYTES = BIG_ENDIAN ? LengthPrefixedWriter::bigEndianIntToBytes :  LengthPrefixedWriter::littleEndianIntToBytes;
    }

    public boolean isBigEndian(){
        return BIG_ENDIAN;
    }

    /**
     * Writes a length-prefixed string to the stream. The length is specified as a 
     * 32-bit unsigned integer in the writer's endian format.
     * 
     * @param string The string to write to the stream.
     * @throws IOException If an I/O error occurs.
     */
    public void writeString(String string) throws IOException{
        byte[] stringBytes = string.getBytes();

        int stringLength = stringBytes.length;
        write(INT_TO_BYTES.apply(stringLength));

        write(stringBytes);
    }

    /**
     * Converts an integer to a 4-byte array in little-endian format.
     * 
     * @param n The integer to convert.
     * @return The byte array representing the integer in little-endian format.
     */
    private static byte[] littleEndianIntToBytes(int n) {
        byte[] bytes = new byte[4];
        bytes[0] = (byte) (n & 0xFF);
        bytes[1] = (byte) ((n >> 8) & 0xFF);
        bytes[2] = (byte) ((n >> 16) & 0xFF);
        bytes[3] = (byte) ((n >> 24) & 0xFF);
        return bytes;
    }

    /**
     * Converts an integer to a 4-byte array in big-endian format.
     * 
     * @param n The integer to convert.
     * @return The byte array representing the integer in big-endian format.
     */
    private static byte[] bigEndianIntToBytes(int n) {
        byte[] bytes = new byte[4];
        bytes[3] = (byte) (n & 0xFF);
        bytes[2] = (byte) ((n >> 8) & 0xFF);
        bytes[1] = (byte) ((n >> 16) & 0xFF);
        bytes[0] = (byte) ((n >> 24) & 0xFF);
        return bytes;
    }
    
}

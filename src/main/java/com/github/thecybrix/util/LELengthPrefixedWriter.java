package com.github.thecybrix.util;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;

public class LELengthPrefixedWriter extends BufferedOutputStream {

    /**
     * Constructs a new LELengthPrefixedWriter with the specified output stream.
     * 
     * @param out The output stream to write to.
     */
    public LELengthPrefixedWriter(OutputStream out) {
        super(out);
    }

    /**
     * Constructs a new LELengthPrefixedWriter with the specified output stream
     * and buffer size.
     * 
     * @param out The output stream to write to.
     * @param size The buffer size.
     */
    public LELengthPrefixedWriter(OutputStream out, int size) {
        super(out, size);
    }

    /**
     * Writes a length-prefixed string to the stream. The length is specified as a 
     * 32-bit unsigned integer in little-endian format.
     * 
     * @param string The string to write to the stream.
     * @throws IOException If an I/O error occurs.
     */
    public void writeString(String string) throws IOException{
        byte[] stringBytes = string.getBytes();

        int stringLength = stringBytes.length;
        write(intToBytes(stringLength));

        write(stringBytes);
    }

    /**
     * Converts an integer to a 4-byte array in little-endian format.
     * 
     * @param n The integer to convert.
     * @return The byte array representing the integer in little-endian format.
     */
    private static byte[] intToBytes(int n) {
        byte[] bytes = new byte[4];
        bytes[0] = (byte) (n & 0xFF);
        bytes[1] = (byte) ((n >> 8) & 0xFF);
        bytes[2] = (byte) ((n >> 16) & 0xFF);
        bytes[3] = (byte) ((n >> 24) & 0xFF);
        return bytes;
    }
    
}

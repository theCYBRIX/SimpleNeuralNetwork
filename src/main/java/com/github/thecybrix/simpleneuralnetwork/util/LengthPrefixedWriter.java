package com.github.thecybrix.simpleneuralnetwork.util;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;

public class LengthPrefixedWriter extends BufferedOutputStream {
    private boolean bigEndian;

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
        this.bigEndian = bigEndian;
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
        this.bigEndian = bigEndian;
    }

    public boolean isBigEndian(){
        return bigEndian;
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
        write(EndianConverter.intToBytes(stringLength, bigEndian));

        write(stringBytes);
    }
    
}

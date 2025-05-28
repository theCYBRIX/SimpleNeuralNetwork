package com.github.thecybrix.simpleneuralnetwork.util;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.SocketException;

public class LengthPrefixedReader extends BufferedInputStream {
    private boolean bigEndian;

    
    /**
     * Constructs a new LengthPrefixedReader with the specified input stream.
     * 
     * @param in The input stream to read from.
     */
    public LengthPrefixedReader(InputStream in) {
        this(in, true);
    }


    /**
     * Constructs a new LengthPrefixedReader with the specified input stream
     * and buffer size.
     * 
     * @param in The input stream to read from.
     * @param size The buffer size.
     */
    public LengthPrefixedReader(InputStream in, int size) {
        this(in, size, true);
    }


    /**
     * Constructs a new LengthPrefixedReader with the specified input stream.
     * 
     * @param in The input stream to read from.
     * @param bigEndian Weather to interpret the length prefix as big-endian.
     */
    public LengthPrefixedReader(InputStream in, boolean bigEndian) {
        super(in);
        this.bigEndian = bigEndian;
    }
    

    /**
     * Constructs a new LengthPrefixedReader with the specified input stream
     * and buffer size.
     * 
     * @param in The input stream to read from.
     * @param size The buffer size.
     * @param bigEndian Weather to interpret the length prefix as big-endian.
     */
    public LengthPrefixedReader(InputStream in, int size, boolean bigEndian) {
        super(in, size);
        this.bigEndian = bigEndian;
    }
    

    public void setBigEndian(boolean bigEndian) {
        this.bigEndian = bigEndian;
    }


    public boolean isBigEndian() {
        return bigEndian;
    }


    /**
     * Reads a length-prefixed string from the stream. The length is specified as a 
     * 32-bit unsigned integer in the reader's endian format.
     * 
     * @return The string read from the stream.
     * @throws SocketException If end of stream is reached before getting the requested number of bytes.
     * @throws IOException If an IOException occurs.
     */
    public String readString() throws SocketException, IOException{
        int length = readLengthPrefix();
        return readString(length);
    }


    /**
     * Reads a 32-bit unsigned integer in the reader's endian format from the stream.
     * 
     * @return An integer indicating the number of string bytes to follow from the stream.
     * @throws SocketException If end of stream is reached before getting the requested number of bytes.
     * @throws IOException If an IOException occurs.
     */
    public int readLengthPrefix() throws SocketException, IOException{
        byte[] lengthBytes = readBytes(4);
        return EndianConverter.bytesToInt(lengthBytes, 0, bigEndian);
    }


    /**
     * Reads a the specified number of bytes from the stream and returns them as a String.
     * 
     * @return A String consisting of the specified number of bytes from the stream.
     * @throws SocketException If end of stream is reached before getting the requested number of bytes.
     * @throws IOException If an IOException occurs.
     */
    public String readString(int length) throws SocketException, IOException{
        byte[] stringBytes = readBytes(length);
        return new String(stringBytes);
    }


    /**
     * Reads the requested number of bytes from the stream.
     * @param stream The stream from which to read.
     * @param numBytes The number of bytes to read.
     * @return An array containing {@code numBytes} bytes from the specified InputStream.
     * @throws SocketException If end of stream is reached before getting the requested number of bytes.
     * @throws IOException If an IOException occurs.
     */
    private byte[] readBytes(int numBytes) throws SocketException, IOException{
        byte[] bytes = new byte[numBytes];
        int totalBytesReceived = 0;
        do {
            int bytesReceived = read(bytes, totalBytesReceived, numBytes - totalBytesReceived);
            if(bytesReceived < 0) throw new SocketException("End of stream reached before receiving requested number of bytes.");
            totalBytesReceived += bytesReceived;
        } while(totalBytesReceived < numBytes);

        return bytes;
    }
    
}

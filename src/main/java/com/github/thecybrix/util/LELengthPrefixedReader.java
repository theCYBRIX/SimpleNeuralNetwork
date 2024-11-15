package com.github.thecybrix.util;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.SocketException;

public class LELengthPrefixedReader extends BufferedInputStream {
    /**
     * Constructs a new LELengthPrefixedReader with the specified input stream.
     * 
     * @param in The input stream to read from.
     */
    public LELengthPrefixedReader(InputStream in) {
        super(in);
    }
    
    /**
     * Constructs a new LELengthPrefixedReader with the specified input stream
     * and buffer size.
     * 
     * @param in The input stream to read from.
     * @param size The buffer size.
     */
    public LELengthPrefixedReader(InputStream in, int size) {
        super(in, size);
    }

    /**
     * Reads a length-prefixed string from the stream. The length is specified as a 
     * 32-bit unsigned integer in little-endian format.
     * 
     * @return The string read from the stream.
     * @throws SocketException If end of stream is reached before getting the requested number of bytes.
     * @throws IOException If an IOException occurs.
     */
    public String readString() throws SocketException, IOException{
        byte[] lengthBytes = readBytes(4);
        int length = bytesToInt(lengthBytes);
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

    /**
     * Converts a 4-byte array in little-endian format to an integer.
     * 
     * @param bytes The byte array to convert. Must be exactly 4 bytes long.
     * @return The integer value of the byte array in little-endian format.
     */
    private static int bytesToInt(byte[] bytes) {
        return (bytes[0] & 0xFF) |
            ((bytes[1] & 0xFF) << 8) |
            ((bytes[2] & 0xFF) << 16) |
            ((bytes[3] & 0xFF) << 24);
    }
    
}

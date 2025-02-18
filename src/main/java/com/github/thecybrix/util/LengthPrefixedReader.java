package com.github.thecybrix.util;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.SocketException;
import java.util.function.ToIntFunction;

public class LengthPrefixedReader extends BufferedInputStream {
    final private ToIntFunction<byte[]> BYTES_TO_INT;
    final private boolean BIG_ENDIAN;

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
        BIG_ENDIAN = bigEndian;
        BYTES_TO_INT = BIG_ENDIAN ? LengthPrefixedReader::bigEndianBytesToInt :  LengthPrefixedReader::littleEndianBytesToInt;
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
        BIG_ENDIAN = bigEndian;
        BYTES_TO_INT = BIG_ENDIAN ? LengthPrefixedReader::bigEndianBytesToInt :  LengthPrefixedReader::littleEndianBytesToInt;
    }

    public boolean isBigEndian() {
        return BIG_ENDIAN;
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
        byte[] lengthBytes = readBytes(4);
        int length = BYTES_TO_INT.applyAsInt(lengthBytes);
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
    private static int littleEndianBytesToInt(byte[] bytes) {
        return (bytes[0] & 0xFF) |
            ((bytes[1] & 0xFF) << 8) |
            ((bytes[2] & 0xFF) << 16) |
            ((bytes[3] & 0xFF) << 24);
    }

    /**
     * Converts a 4-byte array in big-endian format to an integer.
     * 
     * @param bytes The byte array to convert. Must be exactly 4 bytes long.
     * @return The integer value of the byte array in big-endian format.
     */
    private static int bigEndianBytesToInt(byte[] bytes) {
        return (bytes[3] & 0xFF) |
            ((bytes[2] & 0xFF) << 8) |
            ((bytes[1] & 0xFF) << 16) |
            ((bytes[0] & 0xFF) << 24);
    }
    
}

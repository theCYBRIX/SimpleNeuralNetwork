package com.github.thecybrix.simpleneuralnetwork.util;

public abstract class EndianConverter {


    /**
     * Converts an integer to a 4-byte array in the specified endian byte order.
     * 
     * @param value The integer to convert.
     * @param bigEndian Weather to use big-endian byte order.
     * @return A byte array representing the integer.
     */
    public static byte[] intToBytes(int value, boolean bigEndian) {
        byte[] bytes = new byte[4];
        for (int i = 0; i < 4; i++) {
            int index = bigEndian ? 3 - i : i;
            bytes[index] = (byte) (value >>> (i * 8));
        }
        return bytes;
    }


    /**
     * Converts a 4-byte array into an integer using the specified endian byte order.
     * 
     * @param bytes The bytes to convert.
     * @param bigEndian Weather to use big-endian byte order.
     * @return The integer represented by the byte array.
     */
    public static int bytesToInt(byte[] bytes, boolean bigEndian) {
        if (bytes.length != 4) {
            throw new IllegalArgumentException("Byte array must be exactly 4 bytes long.");
        }
        int value = 0;
        for (int i = 0; i < 4; i++) {
            int index = bigEndian ? 3 - i : i;
            value |= ((int) bytes[index] & 0xFF) << (i * 8);
        }
        return value;
    }


    /**
     * Converts a long to an 8-byte array in the specified endian byte order.
     * 
     * @param value The long to convert.
     * @param bigEndian Weather to use big-endian byte order.
     * @return A byte array representing the long.
     */
    public static byte[] longToBytes(long value, boolean bigEndian) {
        byte[] bytes = new byte[8];
        for (int i = 0; i < 8; i++) {
            int index = bigEndian ? 7 - i : i;
            bytes[index] = (byte) (value >>> (i * 8));
        }
        return bytes;
    }


    /**
     * Converts an 8-byte array into a long using the specified endian byte order.
     * 
     * @param bytes The bytes to convert.
     * @param bigEndian Weather to use big-endian byte order.
     * @return The long represented by the byte array.
     */
    public static long bytesToLong(byte[] bytes, boolean bigEndian) {
        if (bytes.length != 8) {
            throw new IllegalArgumentException("Byte array must be exactly 8 bytes long.");
        }
        long value = 0;
        for (int i = 0; i < 8; i++) {
            int index = bigEndian ? 7 - i : i;
            value |= ((long) bytes[index] & 0xFFL) << (i * 8);
        }
        return value;
    }


    /**
     * Converts a float to a 4-byte array in the specified endian byte order.
     * 
     * @param value The float to convert.
     * @param bigEndian Weather to use big-endian byte order.
     * @return A byte array representing the float.
     */
    public static byte[] floatToBytes(float value, boolean bigEndian) {
        int intValue = Float.floatToIntBits(value);
        return intToBytes(intValue, bigEndian);
    }


    /**
     * Converts a 4-byte array into an float using the specified endian byte order.
     * 
     * @param bytes The bytes to convert.
     * @param bigEndian Weather to use big-endian byte order.
     * @return The float represented by the byte array.
     */
    public static float bytesToFloat(byte[] bytes, boolean bigEndian) {
        int intValue = bytesToInt(bytes, bigEndian);
        return Float.intBitsToFloat(intValue);
    }


    /**
     * Converts a double to an 8-byte array in the specified endian byte order.
     * 
     * @param value The double to convert.
     * @param bigEndian Weather to use big-endian byte order.
     * @return A byte array representing the double.
     */
    public static byte[] doubleToBytes(double value, boolean bigEndian) {
        long longValue = Double.doubleToLongBits(value);
        return longToBytes(longValue, bigEndian);
    }


    /**
     * Converts an 8-byte array into a double using the specified endian byte order.
     * 
     * @param bytes The bytes to convert.
     * @param bigEndian Weather to use big-endian byte order.
     * @return The double represented by the byte array.
     */
    public static double bytesToDouble(byte[] bytes, boolean bigEndian) {
        long longValue = bytesToLong(bytes, bigEndian);
        return Double.longBitsToDouble(longValue);
    }
    

}

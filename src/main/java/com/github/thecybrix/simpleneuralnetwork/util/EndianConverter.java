package com.github.thecybrix.simpleneuralnetwork.util;

public class EndianConverter {
    public static final int BITS_PER_BYTE = 8;
    public static final int BYTES_PER_SHORT = 2;
    public static final int BYTES_PER_INT = 4;
    public static final int BYTES_PER_FLOAT = 4;
    public static final int BYTES_PER_LONG = 8;
    public static final int BYTES_PER_DOUBLE = 8;

    private final boolean bigEndian;

    public EndianConverter(Endianness endianness) {
        this(endianness == Endianness.BIG_ENDIAN);
    }

    public EndianConverter(boolean bigEndian) {
        this.bigEndian = bigEndian;
    }

    public byte[] charToBytes(char value) {
        return shortToBytes((short) value, bigEndian);
    }

    public void charToBytes(char value, byte[] dest, int offset) {
        shortToBytes((short) value, dest, offset, bigEndian);
    }

    public byte[] shortToBytes(short value) {
        return shortToBytes(value, bigEndian);
    }

    public void shortToBytes(short value, byte[] dest, int offset) {
        shortToBytes(value, dest, offset, bigEndian);
    }

    public byte[] intToBytes(int value) {
        return intToBytes(value, bigEndian);
    }

    public void intToBytes(int value, byte[] dest, int offset) {
        intToBytes(value, dest, offset, bigEndian);
    }

    public byte[] longToBytes(long value) {
        return longToBytes(value, bigEndian);
    }

    public void longToBytes(long value, byte[] dest, int offset) {
        longToBytes(value, dest, offset, bigEndian);
    }

    public byte[] floatToBytes(float value) {
        return floatToBytes(value, bigEndian);
    }

    public void floatToBytes(float value, byte[] dest, int offset) {
        floatToBytes(value, dest, offset, bigEndian);
    }

    public byte[] doubleToBytes(double value) {
        return doubleToBytes(value, bigEndian);
    }

    public void doubleToBytes(double value, byte[] dest, int offset) {
        doubleToBytes(value, dest, offset, bigEndian);
    }

    public char bytesToChar(byte[] bytes) {
        return (char) (bytesToShort(bytes, 0, bigEndian) & 0xFFFF);
    }

    public char bytesToChar(byte[] bytes, int offset) {
        return bytesToChar(bytes, offset, bigEndian);
    }

    public short bytesToShort(byte[] bytes) {
        return bytesToShort(bytes, 0, bigEndian);
    }

    public short bytesToShort(byte[] bytes, int offset) {
        return bytesToShort(bytes, offset, bigEndian);
    }

    public int bytesToInt(byte[] bytes) {
        return bytesToInt(bytes, 0, bigEndian);
    }

    public int bytesToInt(byte[] bytes, int offset) {
        return bytesToInt(bytes, offset, bigEndian);
    }

    public long bytesToLong(byte[] bytes) {
        return bytesToLong(bytes, 0, bigEndian);
    }

    public long bytesToLong(byte[] bytes, int offset) {
        return bytesToLong(bytes, offset, bigEndian);
    }

    public float bytesToFloat(byte[] bytes) {
        return bytesToFloat(bytes, 0, bigEndian);
    }

    public float bytesToFloat(byte[] bytes, int offset) {
        return bytesToFloat(bytes, offset, bigEndian);
    }

    public double bytesToDouble(byte[] bytes) {
        return bytesToDouble(bytes, 0, bigEndian);
    }

    public double bytesToDouble(byte[] bytes, int offset) {
        return bytesToDouble(bytes, offset, bigEndian);
    }
    
    public static byte[] shortToBytes(short value, boolean bigEndian) {
        byte[] bytes = new byte[BYTES_PER_SHORT];
        shortToBytes(value, bytes, 0, bigEndian);
        return bytes;
    }

    public static byte[] intToBytes(int value, boolean bigEndian) {
        byte[] bytes = new byte[BYTES_PER_INT];
        intToBytes(value, bytes, 0, bigEndian);
        return bytes;
    }

    public static byte[] longToBytes(long value, boolean bigEndian) {
        byte[] bytes = new byte[BYTES_PER_LONG];
        longToBytes(value, bytes, 0, bigEndian);
        return bytes;
    }

    public static byte[] floatToBytes(float value, boolean bigEndian) {
        byte[] bytes = new byte[BYTES_PER_FLOAT];
        floatToBytes(value, bytes, 0, bigEndian);
        return bytes;
    }
    
    public static byte[] doubleToBytes(double value, boolean bigEndian) {
        byte[] bytes = new byte[BYTES_PER_DOUBLE];
        doubleToBytes(value, bytes, 0, bigEndian);
        return bytes;
    }
    
    public static void shortToBytes(short value, byte[] dest, int offset, boolean bigEndian) {
        for (int i = 0; i < BYTES_PER_SHORT; i++) {
            int index = bigEndian ? (BYTES_PER_SHORT - 1) - i : i;
            dest[index + offset] = (byte) (value >>> (i * BITS_PER_BYTE));
        }
    }

    public static void intToBytes(int value, byte[] dest, int offset, boolean bigEndian) {
        for (int i = 0; i < BYTES_PER_INT; i++) {
            int index = bigEndian ? (BYTES_PER_INT - 1) - i : i;
            dest[index + offset] = (byte) (value >>> (i * BITS_PER_BYTE));
        }
    }

    public static void longToBytes(long value, byte[] dest, int offset, boolean bigEndian) {
        for (int i = 0; i < BYTES_PER_LONG; i++) {
            int index = bigEndian ? (BYTES_PER_LONG - 1) - i : i;
            dest[index + offset] = (byte) (value >>> (i * BITS_PER_BYTE));
        }
    }

    public static void floatToBytes(float value, byte[] dest, int offset, boolean bigEndian) {
        int intValue = Float.floatToIntBits(value);
        intToBytes(intValue, dest, offset, bigEndian);
    }
    
    public static void doubleToBytes(double value, byte[] dest, int offset, boolean bigEndian) {
        long longValue = Double.doubleToLongBits(value);
        longToBytes(longValue, dest, offset, bigEndian);
    }
    
    public static int bytesToInt(byte[] bytes, int offset, boolean bigEndian) {
        if (bytes == null || bytes.length < offset + BYTES_PER_INT) {
            throw new IllegalArgumentException("Byte array must contain at least 4 bytes from the given offset.");
        }
        int value = 0;
        for (int i = 0; i < BYTES_PER_INT; i++) {
            int index = offset + (bigEndian ? (BYTES_PER_INT - 1) - i : i);
            value |= (bytes[index] & 0xFF) << (i * BITS_PER_BYTE);
        }
        return value;
    }

    public static int bytesToInt(byte[] bytes, boolean bigEndian) {
        return bytesToInt(bytes, 0, bigEndian);
    }

    public static short bytesToShort(byte[] bytes, int offset, boolean bigEndian) {
        if (bytes == null || bytes.length < offset + 2) {
            throw new IllegalArgumentException("Byte array must contain at least 2 bytes from the given offset.");
        }
        int value = 0;
        for (int i = 0; i < BYTES_PER_SHORT; i++) {
            int index = offset + (bigEndian ? (BYTES_PER_SHORT - 1) - i : i);
            value |= (bytes[index] & 0xFF) << (i * BITS_PER_BYTE);
        }
        return (short) value;
    }

    public static short bytesToShort(byte[] bytes, boolean bigEndian) {
        return bytesToShort(bytes, 0, bigEndian);
    }

    public static long bytesToLong(byte[] bytes, int offset, boolean bigEndian) {
        if (bytes == null || bytes.length < offset + BYTES_PER_LONG) {
            throw new IllegalArgumentException("Byte array must contain at least " + BYTES_PER_LONG + " bytes from the given offset.");
        }
        long value = 0;
        for (int i = 0; i < BYTES_PER_LONG; i++) {
            int index = offset + (bigEndian ? (BYTES_PER_LONG - 1) - i : i);
            value |= ((long) bytes[index] & 0xFFL) << (i * BITS_PER_BYTE);
        }
        return value;
    }

    public static long bytesToLong(byte[] bytes, boolean bigEndian) {
        return bytesToLong(bytes, 0, bigEndian);
    }

    public static float bytesToFloat(byte[] bytes, int offset, boolean bigEndian) {
        int intBits = bytesToInt(bytes, offset, bigEndian);
        return Float.intBitsToFloat(intBits);
    }

    public static float bytesToFloat(byte[] bytes, boolean bigEndian) {
        return bytesToFloat(bytes, 0, bigEndian);
    }

    public static double bytesToDouble(byte[] bytes, int offset, boolean bigEndian) {
        long longBits = bytesToLong(bytes, offset, bigEndian);
        return Double.longBitsToDouble(longBits);
    }

    public static double bytesToDouble(byte[] bytes, boolean bigEndian) {
        return bytesToDouble(bytes, 0, bigEndian);
    }

    public static char bytesToChar(byte[] bytes, int offset, boolean bigEndian) {
        return (char) bytesToShort(bytes, offset, bigEndian);
    }
    
}

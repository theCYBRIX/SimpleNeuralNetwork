import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.function.BiFunction;
import java.util.function.ToIntBiFunction;
import java.util.function.ToLongBiFunction;

import com.github.thecybrix.simpleneuralnetwork.util.EndianConverter;
import com.github.thecybrix.simpleneuralnetwork.util.Endianness;

public class BinaryConversionTest {

    // Function pairs for each data type
    private BiFunction<byte[], Boolean, Short> bytesToShort;
    private BiFunction<Short, Boolean, byte[]> shortToBytes;

    private BiFunction<byte[], Boolean, Character> bytesToChar;
    private BiFunction<Character, Boolean, byte[]> charToBytes;

    private ToIntBiFunction<byte[], Boolean> bytesToInt;
    private BiFunction<Integer, Boolean, byte[]> intToBytes;

    private ToLongBiFunction<byte[], Boolean> bytesToLong;
    private BiFunction<Long, Boolean, byte[]> longToBytes;

    private BiFunction<byte[], Boolean, Float> bytesToFloat;
    private BiFunction<Float, Boolean, byte[]> floatToBytes;

    private BiFunction<byte[], Boolean, Double> bytesToDouble;
    private BiFunction<Double, Boolean, byte[]> doubleToBytes;
    
    public static void main(String[] args) {
        BinaryConversionTest tester = new BinaryConversionTest();

        tester.setShortConverters(
            EndianConverter::bytesToShort,
            EndianConverter::shortToBytes
        );

        tester.setIntConverters(
            EndianConverter::bytesToInt,
            EndianConverter::intToBytes
        );

        tester.setLongConverters(
            EndianConverter::bytesToLong,
            EndianConverter::longToBytes
        );

        tester.setFloatConverters(
            EndianConverter::bytesToFloat,
            EndianConverter::floatToBytes
        );

        tester.setDoubleConverters(
            EndianConverter::bytesToDouble,
            EndianConverter::doubleToBytes
        );

        EndianConverter littleEndianConverter = new EndianConverter(Endianness.LITTLE_ENDIAN);
        EndianConverter bigEndianConverter = new EndianConverter(Endianness.BIG_ENDIAN);

        tester.setCharConverters(
            (bytes, bigEndian) -> bigEndian ? bigEndianConverter.bytesToChar(bytes) : littleEndianConverter.bytesToChar(bytes),
            (bytes, bigEndian) -> bigEndian ? bigEndianConverter.charToBytes(bytes) : littleEndianConverter.charToBytes(bytes)
        );

        // Run tests
        tester.runAllTests();
    }
    
    public void setShortConverters(BiFunction<byte[], Boolean, Short> bytesToShort, BiFunction<Short, Boolean, byte[]> shortToBytes) {
        this.bytesToShort = bytesToShort;
        this.shortToBytes = shortToBytes;
    }

    public void setCharConverters(BiFunction<byte[], Boolean, Character> bytesToChar, BiFunction<Character, Boolean, byte[]> charToBytes) {
        this.bytesToChar = bytesToChar;
        this.charToBytes = charToBytes;
    }

    public void setIntConverters(ToIntBiFunction<byte[], Boolean> bytesToInt,
                                 BiFunction<Integer, Boolean, byte[]> intToBytes) {
        this.bytesToInt = bytesToInt;
        this.intToBytes = intToBytes;
    }

    public void setLongConverters(ToLongBiFunction<byte[], Boolean> bytesToLong,
                                  BiFunction<Long, Boolean, byte[]> longToBytes) {
        this.bytesToLong = bytesToLong;
        this.longToBytes = longToBytes;
    }

    public void setFloatConverters(BiFunction<byte[], Boolean, Float> bytesToFloat,
                                   BiFunction<Float, Boolean, byte[]> floatToBytes) {
        this.bytesToFloat = bytesToFloat;
        this.floatToBytes = floatToBytes;
    }

    public void setDoubleConverters(BiFunction<byte[], Boolean, Double> bytesToDouble,
                                    BiFunction<Double, Boolean, byte[]> doubleToBytes) {
        this.bytesToDouble = bytesToDouble;
        this.doubleToBytes = doubleToBytes;
    }

    private <T> void testConversion(String label, T value, BiFunction<T, Boolean, byte[]> toBytes, BiFunction<byte[], Boolean, T> fromBytes, Boolean bigEndian) {
        byte[] bytes = toBytes.apply(value, bigEndian);
        byte[] reference = getReferenceBytes(value, bigEndian);
        T result = fromBytes.apply(bytes, bigEndian);
        boolean match = value.equals(result);
        boolean byteMatch = validateBytesEqual(bytes, reference);

        printTestResult(label, bigEndian, value, result, match, byteMatch);
    }

    private void testIntConversion(int value, Boolean bigEndian) {
        byte[] bytes = intToBytes.apply(value, bigEndian);
        byte[] reference = getReferenceBytes(value, bigEndian);
        int result = bytesToInt.applyAsInt(bytes, bigEndian);
        boolean match = value == result;
        boolean byteMatch = validateBytesEqual(bytes, reference);

       printTestResult("INT", bigEndian, value, result, match, byteMatch);
    }

    private void testLongConversion(long value, Boolean bigEndian) {
        byte[] bytes = longToBytes.apply(value, bigEndian);
        byte[] reference = getReferenceBytes(value, bigEndian);
        long result = bytesToLong.applyAsLong(bytes, bigEndian);
        boolean match = value == result;
        boolean byteMatch = validateBytesEqual(bytes, reference);

        printTestResult("LONG", bigEndian, value, result, match, byteMatch);
    }

    private void testShortConversion(short value, Boolean bigEndian) {
        byte[] bytes = shortToBytes.apply(value, bigEndian);
        byte[] reference = getReferenceBytes(value, bigEndian);
        short result = bytesToShort.apply(bytes, bigEndian);
        boolean match = value == result;
        boolean byteMatch = validateBytesEqual(bytes, reference);

        printTestResult("SHORT", bigEndian, value, result, match, byteMatch);
    }

    private void testCharConversion(char value, Boolean bigEndian) {
        byte[] bytes = charToBytes.apply(value, bigEndian);
        byte[] reference = getReferenceBytes(value, bigEndian);
        char result = bytesToChar.apply(bytes, bigEndian);
        boolean match = value == result;
        boolean byteMatch = validateBytesEqual(bytes, reference);

        printTestResult("CHAR", bigEndian, value, result, match, byteMatch);
    }

    private void printTestResult(String type, boolean bigEndian, Object input, Object output, boolean match, boolean byteMatch) {
        System.out.printf("\u001B[95m%s\u001B[0m [\u001B[94m%s endian\u001B[0m] - Input: %s, Output: %s, Match: %s, BytesMatch: %s%n",
                type,
                bigEndian ? "Big" : "Little",
                input,
                output,
                match ? "\u001B[92mtrue\u001B[0m" : "\u001B[91mfalse\u001B[0m",
                byteMatch ? "\u001B[92mtrue\u001B[0m" : "\u001B[91mfalse\u001B[0m");
    }

    private boolean validateBytesEqual(byte[] actual, byte[] expected) {
        if (actual.length != expected.length) return false;
        for (int i = 0; i < actual.length; i++) {
            if (actual[i] != expected[i]) return false;
        }
        return true;
    }
    
    private byte[] getReferenceBytes(Object value, boolean bigEndian) {
        ByteBuffer buffer = null;

        if (value instanceof Short) {
            buffer = ByteBuffer.allocate(2);
            buffer.order(bigEndian ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN);
            buffer.putShort((Short) value);
        } else if (value instanceof Character) {
            buffer = ByteBuffer.allocate(2);
            buffer.order(bigEndian ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN);
            buffer.putChar((Character) value);
        } else if (value instanceof Integer) {
            buffer = ByteBuffer.allocate(4);
            buffer.order(bigEndian ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN);
            buffer.putInt((Integer) value);
        } else if (value instanceof Long) {
            buffer = ByteBuffer.allocate(8);
            buffer.order(bigEndian ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN);
            buffer.putLong((Long) value);
        } else if (value instanceof Float) {
            buffer = ByteBuffer.allocate(4);
            buffer.order(bigEndian ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN);
            buffer.putFloat((Float) value);
        } else if (value instanceof Double) {
            buffer = ByteBuffer.allocate(8);
            buffer.order(bigEndian ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN);
            buffer.putDouble((Double) value);
        }

        return buffer != null ? buffer.array() : null;
    }

    public void runAllTests() {
        runSelectedTests(true, true, true, true, true, true);
    }

    public void runSelectedTests(boolean testInt, boolean testLong, boolean testFloat, boolean testDouble, boolean testShort, boolean testChar){
        System.out.println("Running selected tests...");

        int[] testInts = {0, 1, -1, Integer.MAX_VALUE, Integer.MIN_VALUE};
        long[] testLongs = {0L, 1L, -1L, Long.MAX_VALUE, Long.MIN_VALUE};
        float[] testFloats = {0.0f, 1.5f, -2.3f, Float.MAX_VALUE, Float.MIN_VALUE};
        double[] testDoubles = {0.0, 3.14, -9.81, Double.MAX_VALUE, Double.MIN_VALUE};
        short[] testShorts = {0, 1, -1, Short.MAX_VALUE, Short.MIN_VALUE};
        char[] testChars = {'\u0000', 'A', 'z', '\uFFFF'};

        for (boolean endian : new boolean[]{false, true}) {
            if (testInt) {
                for (int i : testInts) testIntConversion(i, endian);
            }
            if (testLong) {
                for (long l : testLongs) testLongConversion(l, endian);
            }
            if (testFloat) {
                for (float f : testFloats) testConversion("FLOAT", f, floatToBytes, bytesToFloat, endian);
            }
            if (testDouble) {
                for (double d : testDoubles) testConversion("DOUBLE", d, doubleToBytes, bytesToDouble, endian);
            }
            if (testShort) {
                for (short s : testShorts) testShortConversion(s, endian);
            }
            if (testChar) {
                for (char c : testChars) testCharConversion(c, endian);
            }
        }
    }
}


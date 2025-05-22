import java.util.function.BiFunction;
import java.util.function.ToIntBiFunction;
import java.util.function.ToLongBiFunction;

import com.github.thecybrix.simpleneuralnetwork.util.EndianConverter;

public class BinaryConversionTest {

    // Function pairs for each data type
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

        // Example implementations using ByteBuffer (Java standard)
        // tester.setIntConverters(
        //     (bytes, bigEndian) -> {
        //         java.nio.ByteBuffer buffer = java.nio.ByteBuffer.wrap(bytes);
        //         buffer.order(bigEndian ? java.nio.ByteOrder.LITTLE_ENDIAN : java.nio.ByteOrder.BIG_ENDIAN);
        //         return buffer.getInt();
        //     },
        //     (value, bigEndian) -> {
        //         java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate(4);
        //         buffer.order(bigEndian ? java.nio.ByteOrder.LITTLE_ENDIAN : java.nio.ByteOrder.BIG_ENDIAN);
        //         buffer.putInt(value);
        //         return buffer.array();
        //     }
        // );

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

        // Run tests
        tester.runAllTests();
        // tester.runSelectedTests(true, false, false, false);
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

    private <T> void testConversion(String label, T value, BiFunction<T, Boolean, byte[]> toBytes,
                                    BiFunction<byte[], Boolean, T> fromBytes, Boolean bigEndian) {
        byte[] bytes = toBytes.apply(value, bigEndian);
        T result = fromBytes.apply(bytes, bigEndian);
        boolean success = value.equals(result);
        System.out.printf("\u001B[95m%s\u001B[0m [\u001B[94m%s endian\u001B[0m] - Input: %s, Output: %s, Match: %s%n",
            label, bigEndian ? "Little" : "Big", value, result,
            success ? "\u001B[92mtrue\u001B[0m" : "\u001B[91mfalse\u001B[0m");

    }

    private void testIntConversion(int value, Boolean bigEndian) {
        byte[] bytes = intToBytes.apply(value, bigEndian);
        int result = bytesToInt.applyAsInt(bytes, bigEndian);
        boolean success = value == result;
        System.out.printf("\u001B[95mINT\u001B[0m [\u001B[94m%s endian\u001B[0m] - Input: %d, Output: %d, Match: %s%n",
            bigEndian ? "Little" : "Big", value, result,
            success ? "\u001B[92mtrue\u001B[0m" : "\u001B[91mfalse\u001B[0m");

    }

    private void testLongConversion(long value, Boolean bigEndian) {
        byte[] bytes = longToBytes.apply(value, bigEndian);
        long result = bytesToLong.applyAsLong(bytes, bigEndian);
        boolean success = value == result;
        System.out.printf("\u001B[95mLONG\u001B[0m [\u001B[94m%s endian\u001B[0m] - Input: %d, Output: %d, Match: %s%n",
            bigEndian ? "Little" : "Big", value, result,
            success ? "\u001B[92mtrue\u001B[0m" : "\u001B[91mfalse\u001B[0m");

    }

    public void runAllTests() {
        runSelectedTests(true, true, true, true);
    }

    public void runSelectedTests(boolean testInt, boolean testLong, boolean testFloat, boolean testDouble) {
        System.out.println("Running selected tests...");

        int[] testInts = {0, 1, -1, Integer.MAX_VALUE, Integer.MIN_VALUE};
        long[] testLongs = {0L, 1L, -1L, Long.MAX_VALUE, Long.MIN_VALUE};
        float[] testFloats = {0.0f, 1.5f, -2.3f, Float.MAX_VALUE, Float.MIN_VALUE};
        double[] testDoubles = {0.0, 3.14, -9.81, Double.MAX_VALUE, Double.MIN_VALUE};

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
        }
    }
}


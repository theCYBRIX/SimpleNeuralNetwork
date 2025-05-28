import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Random;

import com.github.thecybrix.simpleneuralnetwork.util.EndianAwareInputStream;
import com.github.thecybrix.simpleneuralnetwork.util.EndianAwareOutputStream;
import com.github.thecybrix.simpleneuralnetwork.util.EndianConverter;
import com.github.thecybrix.simpleneuralnetwork.util.Endianness;
import com.github.thecybrix.simpleneuralnetwork.util.StopWatch;

public class EndianConversionBenchmark {

    private static final int SAMPLES = 10_000_000;
    private static final Endianness ENDIANNESS = Endianness.LITTLE_ENDIAN;

    private static final String ANSI_GREEN = "\u001B[32m";
    private static final String ANSI_RED = "\u001B[31m";
    private static final String ANSI_RESET = "\u001B[0m";

    private static void printResult(String label, double time1, double time2) {
        String fastColor = time1 <= time2 ? ANSI_GREEN : ANSI_RED;
        String slowColor = time1 > time2 ? ANSI_GREEN : ANSI_RED;

        System.out.printf("%-10s Static: %s%.3fms%s  Stream: %s%.3fms%s%n",
                label,
                fastColor, time1, ANSI_RESET,
                slowColor, time2, ANSI_RESET
        );
    }

    public static void main(String[] args) throws IOException {
        Random rand = new Random(42);
        EndianConverter converter = new EndianConverter(ENDIANNESS);

        benchmarkShort(rand, converter);
        benchmarkChar(rand, converter);
        benchmarkInt(rand, converter);
        benchmarkLong(rand, converter);
        benchmarkFloat(rand, converter);
        benchmarkDouble(rand, converter);
    }

    @SuppressWarnings("resource")
    private static void benchmarkShort(Random rand, EndianConverter converter) throws IOException {
        short[] data = new short[SAMPLES];
        for (int i = 0; i < SAMPLES; i++) {
            data[i] = (short) rand.nextInt();
        }

        StopWatch sw = new StopWatch();

        // Static
        sw.start();
        ByteArrayOutputStream byteOutputStream = new ByteArrayOutputStream();
        for (short value : data) {
            byteOutputStream.write(converter.shortToBytes(value));
        }
        ByteArrayInputStream byteInputStream = new ByteArrayInputStream(byteOutputStream.toByteArray());
        for (int i = 0; i < SAMPLES; i++) {
            byte[] buffer = byteInputStream.readNBytes(2);
            converter.bytesToShort(buffer);
        }
        sw.stop();
        double staticTime = sw.getMillisExact();

        // Stream
        sw.start();
        ByteArrayOutputStream streamOut = new ByteArrayOutputStream();
        EndianAwareOutputStream out = new EndianAwareOutputStream(streamOut, ENDIANNESS);
        for (short value : data) {
            out.writeShort(value);
        }
        ByteArrayInputStream streamIn = new ByteArrayInputStream(streamOut.toByteArray());
        EndianAwareInputStream in = new EndianAwareInputStream(streamIn, ENDIANNESS);
        for (int i = 0; i < SAMPLES; i++) {
            in.readShort();
        }
        sw.stop();
        double streamTime = sw.getMillisExact();

        printResult("short", staticTime, streamTime);
    }

    @SuppressWarnings("resource")
    private static void benchmarkChar(Random rand, EndianConverter converter) throws IOException {
        char[] data = new char[SAMPLES];
        for (int i = 0; i < SAMPLES; i++) {
            data[i] = (char) rand.nextInt(65536);
        }

        StopWatch sw = new StopWatch();

        // Static
        sw.start();
        ByteArrayOutputStream byteOutputStream = new ByteArrayOutputStream();
        for (char value : data) {
            byteOutputStream.write(converter.charToBytes(value));
        }
        ByteArrayInputStream byteInputStream = new ByteArrayInputStream(byteOutputStream.toByteArray());
        for (int i = 0; i < SAMPLES; i++) {
            byte[] buffer = byteInputStream.readNBytes(2);
            converter.bytesToChar(buffer);
        }
        sw.stop();
        double staticTime = sw.getMillisExact();

        // Stream
        sw.start();
        ByteArrayOutputStream streamOut = new ByteArrayOutputStream();
        EndianAwareOutputStream out = new EndianAwareOutputStream(streamOut, ENDIANNESS);
        for (char value : data) {
            out.writeChar(value);
        }
        ByteArrayInputStream streamIn = new ByteArrayInputStream(streamOut.toByteArray());
        EndianAwareInputStream in = new EndianAwareInputStream(streamIn, ENDIANNESS);
        for (int i = 0; i < SAMPLES; i++) {
            in.readChar();
        }
        sw.stop();
        double streamTime = sw.getMillisExact();

        printResult("char", staticTime, streamTime);
    }

    @SuppressWarnings("resource")
    private static void benchmarkInt(Random rand, EndianConverter converter) throws IOException {
        int[] data = new int[SAMPLES];
        for (int i = 0; i < SAMPLES; i++) {
            data[i] = rand.nextInt();
        }

        StopWatch sw = new StopWatch();

        // Static
        sw.start();
        ByteArrayOutputStream byteOutputStream = new ByteArrayOutputStream();
        for (int value : data) {
            byteOutputStream.write(converter.intToBytes(value));
        }
        ByteArrayInputStream byteInputStream = new ByteArrayInputStream(byteOutputStream.toByteArray());
        for (int i = 0; i < SAMPLES; i++) {
            byte[] buffer = byteInputStream.readNBytes(4);
            converter.bytesToInt(buffer);
        }
        sw.stop();
        double staticTime = sw.getMillisExact();

        // Stream
        sw.start();
        ByteArrayOutputStream streamOut = new ByteArrayOutputStream();
        EndianAwareOutputStream out = new EndianAwareOutputStream(streamOut, ENDIANNESS);
        for (int value : data) {
            out.writeInt(value);
        }
        ByteArrayInputStream streamIn = new ByteArrayInputStream(streamOut.toByteArray());
        EndianAwareInputStream in = new EndianAwareInputStream(streamIn, ENDIANNESS);
        for (int i = 0; i < SAMPLES; i++) {
            in.readInt();
        }
        sw.stop();
        double streamTime = sw.getMillisExact();

        printResult("int", staticTime, streamTime);
    }

    @SuppressWarnings("resource")
    private static void benchmarkLong(Random rand, EndianConverter converter) throws IOException {
        long[] data = new long[SAMPLES];
        for (int i = 0; i < SAMPLES; i++) {
            data[i] = rand.nextLong();
        }

        StopWatch sw = new StopWatch();

        // Static
        sw.start();
        ByteArrayOutputStream byteOutputStream = new ByteArrayOutputStream();
        for (long value : data) {
            byteOutputStream.write(converter.longToBytes(value));
        }
        ByteArrayInputStream byteInputStream = new ByteArrayInputStream(byteOutputStream.toByteArray());
        for (int i = 0; i < SAMPLES; i++) {
            byte[] buffer = byteInputStream.readNBytes(8);
            converter.bytesToLong(buffer);
        }
        sw.stop();
        double staticTime = sw.getMillisExact();

        // Stream
        sw.start();
        ByteArrayOutputStream streamOut = new ByteArrayOutputStream();
        EndianAwareOutputStream out = new EndianAwareOutputStream(streamOut, ENDIANNESS);
        for (long value : data) {
            out.writeLong(value);
        }
        ByteArrayInputStream streamIn = new ByteArrayInputStream(streamOut.toByteArray());
        EndianAwareInputStream in = new EndianAwareInputStream(streamIn, ENDIANNESS);
        for (int i = 0; i < SAMPLES; i++) {
            in.readLong();
        }
        sw.stop();
        double streamTime = sw.getMillisExact();

        printResult("long", staticTime, streamTime);
    }

    @SuppressWarnings("resource")
    private static void benchmarkFloat(Random rand, EndianConverter converter) throws IOException {
        float[] data = new float[SAMPLES];
        for (int i = 0; i < SAMPLES; i++) {
            data[i] = rand.nextFloat();
        }

        StopWatch sw = new StopWatch();

        // Static
        sw.start();
        ByteArrayOutputStream byteOutputStream = new ByteArrayOutputStream();
        for (float value : data) {
            byteOutputStream.write(converter.floatToBytes(value));
        }
        ByteArrayInputStream byteInputStream = new ByteArrayInputStream(byteOutputStream.toByteArray());
        for (int i = 0; i < SAMPLES; i++) {
            byte[] buffer = byteInputStream.readNBytes(4);
            converter.bytesToFloat(buffer);
        }
        sw.stop();
        double staticTime = sw.getMillisExact();

        // Stream
        sw.start();
        ByteArrayOutputStream streamOut = new ByteArrayOutputStream();
        EndianAwareOutputStream out = new EndianAwareOutputStream(streamOut, ENDIANNESS);
        for (float value : data) {
            out.writeFloat(value);
        }
        ByteArrayInputStream streamIn = new ByteArrayInputStream(streamOut.toByteArray());
        EndianAwareInputStream in = new EndianAwareInputStream(streamIn, ENDIANNESS);
        for (int i = 0; i < SAMPLES; i++) {
            in.readFloat();
        }
        sw.stop();
        double streamTime = sw.getMillisExact();

        printResult("float", staticTime, streamTime);
    }

    @SuppressWarnings("resource")
    private static void benchmarkDouble(Random rand, EndianConverter converter) throws IOException {
        double[] data = new double[SAMPLES];
        for (int i = 0; i < SAMPLES; i++) {
            data[i] = rand.nextDouble();
        }

        StopWatch sw = new StopWatch();

        // Static
        sw.start();
        ByteArrayOutputStream byteOutputStream = new ByteArrayOutputStream();
        for (double value : data) {
            byteOutputStream.write(converter.doubleToBytes(value));
        }
        ByteArrayInputStream byteInputStream = new ByteArrayInputStream(byteOutputStream.toByteArray());
        for (int i = 0; i < SAMPLES; i++) {
            byte[] buffer = byteInputStream.readNBytes(8);
            converter.bytesToDouble(buffer);
        }
        sw.stop();
        double staticTime = sw.getMillisExact();

        // Stream
        sw.start();
        ByteArrayOutputStream streamOut = new ByteArrayOutputStream();
        EndianAwareOutputStream out = new EndianAwareOutputStream(streamOut, ENDIANNESS);
        for (double value : data) {
            out.writeDouble(value);
        }
        ByteArrayInputStream streamIn = new ByteArrayInputStream(streamOut.toByteArray());
        EndianAwareInputStream in = new EndianAwareInputStream(streamIn, ENDIANNESS);
        for (int i = 0; i < SAMPLES; i++) {
            in.readDouble();
        }
        sw.stop();
        double streamTime = sw.getMillisExact();

        printResult("double", staticTime, streamTime);
    }
}

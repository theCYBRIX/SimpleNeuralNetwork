import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class Test2 {
    public static void main(String[] args) {

        System.out.println(13%12);
        int j = 1;
        if(j + 1 == 2) return;

        int numOptions = 500;
        Integer[] numbers = new Integer[numOptions];

        for (int i = 0, next; i < numbers.length; i = next) {
            next = i + 1;
            numbers[i] = Integer.valueOf(next);
        }

        List<Integer> randoms = getRandomLikelyLast(Arrays.asList(numbers), numOptions / 5);
        int leastLikely = 0,
            lessLikely = 0,
            moreLikely = 0,
            mostLikely = 0;

        int firstQuarter = Math.round(numOptions * 0.95f);
        int secondQuarter = Math.round(numOptions * 0.97f);
        int thirdQuarter = Math.round(numOptions * 0.99f);

        for(int number : randoms){
            if(number <= firstQuarter)
                leastLikely++;
            else if(number <= secondQuarter)
                lessLikely++;
            else if(number <= thirdQuarter)
                moreLikely++;
            else
                mostLikely++;
        }

        DecimalFormat decimalFormat = new DecimalFormat("0.00");
        StringBuilder results = new StringBuilder();

        results.append("\nLeast likely: ").append(decimalFormat.format(leastLikely / (float)randoms.size()));
        results.append("\nLess likely: ").append(decimalFormat.format(lessLikely / (float)randoms.size()));
        results.append("\nMore likely: ").append(decimalFormat.format(moreLikely / (float)randoms.size()));
        results.append("\nMost likely: ").append(decimalFormat.format(mostLikely / (float)randoms.size()));

        System.out.println(results.toString());
    }
    

	
    
    private static <T> List<T> getRandomLikelyLast(List<T> selection, int numItems) throws IllegalArgumentException {
        if (numItems < 0 || numItems > selection.size()) {
            throw new IllegalArgumentException("Invalid numItems value");
        }

        List<T> result = new ArrayList<>(numItems);
        ArrayList<Integer> indexes = new ArrayList<>();
        Random random = new Random();

        for (int i = 0; i < selection.size(); i++) {
            indexes.add(i);
        }
		
		int lastIndex = indexes.size() - 1;
		double[] probabilities = new double[selection.size()];
		double totalProbability = 0;
		for (int i = 0; i <= lastIndex; i++) {
			probabilities[i] = Math.pow(2, i);
			totalProbability += probabilities[i];
		}

        for (int i = 0; i < numItems; i++) {
            Integer selectedIndex = null;
			double randomWeight = random.nextDouble() * totalProbability;
			double weightSum = 0;
			for (int j = 0; j <= lastIndex; j++) {
				weightSum += probabilities[j];
				if (randomWeight <= weightSum) {
					selectedIndex = indexes.get(j);
					break;
				}
			}
			if(selectedIndex == null) selectedIndex = indexes.get(lastIndex);

            result.add(selection.get(selectedIndex.intValue()));
        }

        return result;
    }


    static String arrayToString(double[] array){
        StringBuilder out = new StringBuilder();
        
        out.append("[").append(array[0]);
        for(int i = 1; i < array.length; i++)
            out.append(", ").append(array[i]);
        out.append("]");

        return out.toString();
    }
}

package com.github.thecybrix.simpleneuralnetwork.training.evolution;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.OptionalDouble;
import java.util.function.Consumer;

public class TrainingScoreLogger implements Consumer<EvolutionaryTrainer<?>>{

	final private static byte DEF_SCORE_HISTORY_SIZE = 100;
	private int scoreHistorySize = DEF_SCORE_HISTORY_SIZE;
    private int gensWithoutImprovement = 0;

	final private LinkedList<Double> BEST_SCORE_HISTORY = new LinkedList<Double>();
    final private List<Double> READ_ONLY_VIEW = Collections.unmodifiableList(BEST_SCORE_HISTORY);

	private void updateScoreHistory(OptionalDouble bestScore) throws NullPointerException{

		if(bestScore.isPresent()){
			if(BEST_SCORE_HISTORY.isEmpty()){
				gensWithoutImprovement = 0;
				BEST_SCORE_HISTORY.add(bestScore.getAsDouble());
				
			} else {
				double previousBest = BEST_SCORE_HISTORY.getLast();
				double currentBest = bestScore.getAsDouble();
				BEST_SCORE_HISTORY.add(currentBest);
				if(Double.compare(previousBest, currentBest) < 0)
					gensWithoutImprovement = 0;
				else
					gensWithoutImprovement++;
			}
		} else if(BEST_SCORE_HISTORY.size() > 0) {
			BEST_SCORE_HISTORY.add(BEST_SCORE_HISTORY.getLast());
			gensWithoutImprovement++;
		}

		while(BEST_SCORE_HISTORY.size() > scoreHistorySize){
			BEST_SCORE_HISTORY.removeFirst();
		}
	}

	public void reset(){
		BEST_SCORE_HISTORY.clear();
		gensWithoutImprovement = 0;
	}

    @Override
    public void accept(EvolutionaryTrainer<?> trainer) {
        updateScoreHistory(Collections.max(trainer.getPreviousGeneration()).getScore());
    }
 
	public List<Double> getScoreHistory(){
		return READ_ONLY_VIEW;
	}

    public int getGensWithoutImprovement() {
        return gensWithoutImprovement;
    }

	public void setScoreHistorySize(int size) throws IllegalArgumentException {
		if(size < 0) throw new IllegalArgumentException();
		this.scoreHistorySize = size;
	}
    
}

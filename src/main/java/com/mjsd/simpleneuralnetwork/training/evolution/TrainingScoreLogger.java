package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

public class TrainingScoreLogger<T extends Comparable<T>> implements Consumer<EvolutionaryTrainer<?, T>>{

	final private static byte DEF_SCORE_HISTORY_SIZE = 100;
	private int scoreHistorySize = DEF_SCORE_HISTORY_SIZE;
    private int gensWithoutImprovement = 0;

	final private LinkedList<T> BEST_SCORE_HISTORY = new LinkedList<T>();
    final private List<T> READ_ONLY_VIEW = Collections.unmodifiableList(BEST_SCORE_HISTORY);

	private void updateScoreHistory(Optional<T> bestScore) throws NullPointerException{

		if(bestScore.isPresent()){
			if(BEST_SCORE_HISTORY.isEmpty()){
				gensWithoutImprovement = 0;
				BEST_SCORE_HISTORY.add(bestScore.get());
				
			} else {
				T previousBest = BEST_SCORE_HISTORY.getLast();
				T currentBest = bestScore.get();
				BEST_SCORE_HISTORY.add(currentBest);
				if(previousBest.compareTo(currentBest) < 0)
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
    public void accept(EvolutionaryTrainer<?, T> trainer) {
        updateScoreHistory(Collections.max(trainer.getPreviousGeneration()).getScore());
    }
 
	public List<T> getScoreHistory(){
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

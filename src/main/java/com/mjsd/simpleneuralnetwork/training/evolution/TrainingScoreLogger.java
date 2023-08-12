package com.mjsd.simpleneuralnetwork.training.evolution;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

import com.mjsd.simpleneuralnetwork.training.RankedNeuralNetwork;

public class TrainingScoreLogger implements Consumer<EvolutionaryTrainer<? extends RankedNeuralNetwork>>{

	final private static byte DEF_SCORE_HISTORY_SIZE = 100;
	private int scoreHistorySize = DEF_SCORE_HISTORY_SIZE;
    private int gensWithoutImprovement = 0;

	final private LinkedList<Double> BEST_SCORE_HISTORY = new LinkedList<Double>();
    final private List<Double> READ_ONLY_VIEW = Collections.unmodifiableList(BEST_SCORE_HISTORY);

	private void updateScoreHistory(Optional<Double> bestScore){

		if(bestScore.isPresent()){
			if(BEST_SCORE_HISTORY.isEmpty()){
				gensWithoutImprovement = 0;
				BEST_SCORE_HISTORY.add(bestScore.get());
				
			} else {
				Double previousBest = BEST_SCORE_HISTORY.getLast();
				Double currentBest = bestScore.get();
				BEST_SCORE_HISTORY.add(currentBest);
				if(previousBest < currentBest)
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
    public void accept(EvolutionaryTrainer<? extends RankedNeuralNetwork> trainer) {
        updateScoreHistory(trainer.getEcosystem().getBestScore());
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

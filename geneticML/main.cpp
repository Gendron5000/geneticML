#include "util.h"
#include "userRNG.h"
#include "organism.h"
#include "geneticAlgoTrainer.h"

#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>

#include <mlpack/methods/ann/layer/lstm.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/methods/ann/layer/vr_class_reward.hpp>
#include <mlpack/methods/ann/layer/sequential.hpp>
#include <mlpack/methods/ann/layer/recurrent_attention.hpp>
#include <mlpack/methods/ann/layer/recurrent.hpp>
#include <mlpack/methods/ann/layer/multiply_merge.hpp>
#include <mlpack/methods/ann/layer/fast_lstm.hpp>
#include <mlpack/methods/ann/layer/linear_no_bias.hpp>
#include <mlpack/methods/ann/layer/linear.hpp>
#include <mlpack/methods/ann/layer/gru.hpp>
#include <mlpack/methods/ann/layer/glimpse.hpp>
#include <mlpack/methods/ann/layer/dropconnect.hpp>
#include <mlpack/methods/ann/layer/transposed_convolution.hpp>
#include <mlpack/methods/ann/layer/convolution.hpp>
#include <mlpack/methods/ann/layer/concat_performance.hpp>
#include <mlpack/methods/ann/layer/concat.hpp>
#include <mlpack/methods/ann/layer/atrous_convolution.hpp>

#include <functional>

using namespace mlpack::ann;

using RnnType = RNN<SigmoidLayer<>>;

class SudokuSolution
{
using DataType = arma::Mat<int>;
public:
	DataType& Parameters() { return solution; }
	SudokuSolution() 
	{
		solution = DataType(9, 9);
	}

private:
	DataType solution;
};

void RunSudoku()
{
	auto createWeights = [] ()
	{
		return new SudokuSolution();
	};

	auto calculateFitness = [] (SudokuSolution& in_solution)
	{
		double retFitness = 0.0;
		auto& cells = in_solution.Parameters();
		std::set<int> uniqueValues;

		for (int i = 0; i < 9; i++)
		{

			for (auto& it : cells.row(i))
			{
				uniqueValues.insert(it);
			}
			//todo perf test with loop configs

			retFitness += uniqueValues.size();

			uniqueValues.clear();

			for (auto& it : cells.col(i))
			{
				uniqueValues.insert(it);
			}

			retFitness += uniqueValues.size();
			uniqueValues.clear();
		}

		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				arma::subview<int> square = cells.submat(j*3, k*3, arma::SizeMat(3, 3));

				for (auto& it : square)
				{
					uniqueValues.insert(it);
				}
                
                retFitness += uniqueValues.size();
                uniqueValues.clear();
			}
		}

		return retFitness;
	};

    GeneticAlgoTrainer trainer(createWeights, calculateFitness);
	auto& settings = trainer.GetSettings();
	settings.minWeight = 1;
	settings.maxWeight = 9;
    settings.numEpoch = 100000;
	settings.useIntType = true;
    trainer.Run();
}

int main()
{
    const arma::cube trainData = GetInputDataExitOnError(
            "/Users/tkgendro/projects/geneticML/geneticML/trainData.json" );
    
    const arma::cube testData = GetInputDataExitOnError(
            "/Users/tkgendro/projects/geneticML/geneticML/testData.json");
    
    const arma::cube* dataToUse = &trainData;
    
    auto createRNN = []()
    {
        auto pRnn = std::make_unique<RnnType>(1);
        
        int inputs = 1;
        int hiddenSize = 6 * inputs;
        int outputs = 2;
        
        pRnn->Add<LinearNoBias<> >(1, hiddenSize);
        pRnn->Add<LSTM<>>(hiddenSize, hiddenSize);
        pRnn->Add<LSTM<>>(hiddenSize, hiddenSize);
        pRnn->Add<LSTM<>>(hiddenSize, hiddenSize);
        pRnn->Add<LSTM<>>(hiddenSize, outputs);
        pRnn->Add<SigmoidLayer<> >();
        pRnn->Reset();
        
        return pRnn.release();
    };
    
    auto calculateFitness = [&dataToUse] (RnnType& in_rnn)
    {
        double retFitness = 0.0;
        
        arma::cube prediction;
        in_rnn.Predict(*dataToUse, prediction, 1);
        
        long long predictSize = arma::size(prediction)[1];
        double normalizedActualCost = 0.0;
        bool readyToBuy = true;
        
        for (int i = 0; i < predictSize; i++)
        {
            normalizedActualCost += (*dataToUse)(0,i,0);
            
            bool buy = prediction(0,i,0) > 0.5;
            bool sell = prediction(1,i,0) > 0.5;
            
            if (readyToBuy && buy)
            {
                if (!sell)
                {
                    // buy
                    retFitness -= normalizedActualCost;
                    readyToBuy = false;
                }
            }
            else if (!readyToBuy && sell)
            {
                if (!buy)
                {
                    // sell
                    retFitness += normalizedActualCost;
                    readyToBuy = true;
                }
            }
        }
        
        return retFitness;
    };
    
    //GeneticAlgoTrainer<std::function<RnnType*()>, std::function<double(RnnType&, bool)>> trainer((std::function<RnnType*()>(createRNN)), std::function<double(RnnType&, bool)>(calculateFitness));
    GeneticAlgoTrainer trainer(createRNN, calculateFitness);
	auto settings = trainer.GetSettings();
	settings.minWeight = -2.0;
	settings.maxWeight = 2.0;
    trainer.Run();
    
    Log ("fitness from training data:", calculateFitness(*trainer.GetBestPerformer()) );
    dataToUse = &testData;
    Log ("fitness from test data: ", calculateFitness(*trainer.GetBestPerformer()) );

	return 1;
}


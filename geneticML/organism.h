#ifndef ORGANISM_H 
#define ORGANISM_H  

#include <iostream>
#include <unordered_set>

#include "util.h"
#include "userRNG.h"

template <class BaseType>
class Organism
{
using ThisType = Organism<BaseType>;
using pBaseType = std::unique_ptr<BaseType>;

public:
	enum class EvolveType {Random=0, CloneMutation, Child, ChildMutation};

    pBaseType& GetBase(){return pBase;}
    double GetFitness() const {return fitness;}
    void SetFitness(double in_fitness) {fitness = in_fitness;}
    
	Organism() = delete;
	Organism(const ThisType& rhs) = delete;
	Organism(const ThisType&& rhs) = delete;
    ThisType operator()(const ThisType& rhs) = delete;
    ThisType operator()(const ThisType&& rhs) = delete;

	// setup empty recurrent neural net with the in_createFn() call
	Organism(pBaseType&& in_basePtr)
    :pBase(std::forward<pBaseType>(in_basePtr))
	,fitness(0.0)
	,ID(OrganismIndexID++)
	{
		RandomizeWeights();
		Log("creating org, ", pBase->Parameters().size());
	}

	void Evolve(
                const ThisType* parentA,
                const ThisType* parentB,
                double in_mutProb,
                EvolveType in_evolveType )
	{
		ID = OrganismIndexID++;

		if (in_evolveType == EvolveType::Random)
		{
			RandomizeWeights();
		}
		else if (in_evolveType == EvolveType::CloneMutation)
		{
			EvolveCloneWithMutation(parentA, in_mutProb);
		}
		else if (in_evolveType == EvolveType::Child)
		{
			EvolveChildFromParents(parentA, parentB);
		}
		else if (in_evolveType == EvolveType::ChildMutation)
		{
			EvolveChildFromParentsWithMutation(parentA, parentB, in_mutProb);
		}
	}

	void Display()
	{ 
		char buf[100];
		snprintf(buf, 100, "%lld:\t%.3f\t", ID, fitness);
		Log(buf);
	}

	template<typename... Args>
	void DisplayFull(Args&&... args)
	{
		Display();
		((args->Display()), ...);
		/*
		auto& weights = pBase->Parameters();
		for(int i = 0; i < weights.size(); i++)
		{
			printf("\n%.6f", weights[i]);
			((printf("\t%.6f", args->pBase->Parameters()[i]) ), ...);
		}
		*/
	}

private:
	pBaseType pBase;
	double fitness;
	long long ID;

	static long long OrganismIndexID;

	// set all the child weights from one parent or the other (randomly chosen)
	void EvolveChildFromParents(
                          const ThisType* parentA,
                          const ThisType* parentB )
	{
		auto& childWeights = pBase->Parameters();
		auto& parentAWeights = parentA->pBase->Parameters();
		auto& parentBWeights = parentB->pBase->Parameters();

		if ((childWeights.size() != parentAWeights.size()) ||
            (childWeights.size() != parentBWeights.size())    )
		{
			ReportFatalError("error, weights not same");
		}

		auto fiftyFn = UserRNG::GetRngFn(0.5);

		for (int i = 0; i < childWeights.size(); i++)
		{
			// 50/50 chance to get each weight from either parent
			if (fiftyFn())
			{
				childWeights[i] = parentAWeights[i];
			}
			else
			{
				childWeights[i] = parentBWeights[i];
			}
		}
	}

	void EvolveCloneWithMutation(
                        const ThisType* parentA,
                        double in_mutProb )
	{
		pBase->Parameters() = parentA->pBase->Parameters();
		Mutate(in_mutProb);
	}

	void EvolveChildFromParentsWithMutation(
                                      const ThisType* parentA,
                                      const ThisType* parentB,
                                      double in_mutProb )
	{
		EvolveChildFromParents(parentA, parentB);
		Mutate(in_mutProb);
	}

	// randomize all the weights of all the parameters
	void RandomizeWeights()
	{
		auto weightDist = UserRNG::GetRngFn(-2.0, 2.0);
		pBase->Parameters().transform(
            [&weightDist](double x){return weightDist();} );
	}

	void Mutate(double mutationPercentage)
	{
		std::unordered_set<int> mutationIndexes;
		auto& weights = pBase->Parameters();
		auto weightDist = UserRNG::GetRngFn(-2.0, 2.0);

		int numMutations = (double)mutationPercentage * weights.size();
		auto mutationIndexDist = UserRNG::GetRngFn(0, (int) weights.size()-1);

		while (mutationIndexes.size() != numMutations)
		{
			mutationIndexes.insert(mutationIndexDist());
		}

		auto mutateFn = [&weights, &weightDist](int index)
            { weights[index] *= weightDist(); };
        
		std::for_each(mutationIndexes.begin(), mutationIndexes.end(), mutateFn);
	}
};

template <class CreateFn>
long long Organism<CreateFn>::OrganismIndexID = 0;

#endif

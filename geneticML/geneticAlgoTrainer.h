#ifndef GENETICALGOTRAINER_H
#define GENETICALGOTRAINER_H

#include "organism.h"
#include "userRNG.h"
#include "ctpl_stl.h"

template <class CreateFn, class FitnessFn>
class GeneticAlgoTrainer
{
    
using BaseType =
    typename std::remove_pointer<
        typename std::result_of<CreateFn()>::type>::type;
    
using ThreadPool = ctpl::thread_pool;

public:

	struct Settings
	{
        float minWeight = -1.0;
        float maxWeight = 1.0;
		bool useIntType = false;

		float epochDeletePercent = 0.5;
		int numEpoch = 20;
		int numPopulation = 1000;
		int randomWeight = 10;
		int mutationWeight = 20;
		int childWeight = 20;
		int childWithMutationWeight = 50;
        
        float minMutationPercent = 0.0;
        float maxMutationPercent = 0.25;
		
	};

    BaseType* GetBestPerformer() { return nullptr; }//organisms.at(0)->GetBase().get(); }
	Settings& GetSettings() { return settings; }
    
    // forbid copying of any kind
	GeneticAlgoTrainer() = delete;
    GeneticAlgoTrainer(const GeneticAlgoTrainer& rhs) = delete;
    GeneticAlgoTrainer(const GeneticAlgoTrainer&& rhs) = delete;
    GeneticAlgoTrainer operator()(const GeneticAlgoTrainer& rhs) = delete;
    GeneticAlgoTrainer operator()(const GeneticAlgoTrainer&& rhs) = delete;

    GeneticAlgoTrainer(const CreateFn& in_createFn, const FitnessFn& in_fitnessFn)
    :fitnessFn(in_fitnessFn)
    ,createFn(in_createFn)
    ,settings()
    ,workers(std::thread::hardware_concurrency())
    {
    }

	template<class M, class R>
	void _Run(M& mutationFn, R& weightFn)
	{
using OrganismBase = Organism<BaseType, M, R>;
using pOrganism = std::unique_ptr<OrganismBase>;
using Organisms = std::vector<pOrganism>;

		Organisms organisms;

        for (int i = 0; i < settings.numPopulation; i++)
        {
            organisms.emplace_back(
                std::make_unique<OrganismBase>(
                    std::unique_ptr<BaseType>(createFn()),
					mutationFn,
					weightFn ) );
        }
        
		int numOrganismsDel = settings.epochDeletePercent * settings.numPopulation;
		int numOrganismsSave = settings.numPopulation - numOrganismsDel;

		// higher fitness means better odds of being parents
		std::vector<int> parentIndexDistributionWeights(numOrganismsSave);
		std::for_each(
            parentIndexDistributionWeights.begin(),
            parentIndexDistributionWeights.end(),
            [x = numOrganismsSave] (int& weight) mutable {weight = x--;} );

		auto parentIndexDist = UserRNG::GetRngFn(parentIndexDistributionWeights);

		auto mutProbDist = UserRNG::GetRngFn(
				settings.minMutationPercent, settings.maxMutationPercent);

		auto childCreatorDist = UserRNG::GetRngFn(
				std::initializer_list<int>{
					settings.randomWeight, 
					settings.mutationWeight, 
					settings.childWeight, 
					settings.childWithMutationWeight } );

		auto fitnessCmpFn = [](const pOrganism& in_orgA, const pOrganism& in_orgB)
		{
			return in_orgA->GetFitness() > in_orgB->GetFitness();  
		};
        
        auto EvolveThenEval = [this] (
            OrganismBase* child,
            const OrganismBase* parentA,
            const OrganismBase* parentB,
            double mutationProbability,
            typename OrganismBase::EvolveType evolveType )
        {
            child->Evolve(parentA, parentB, evolveType);
            child->SetFitness(fitnessFn(*child->GetBase()));
        };
        
        std::vector<std::future<void>> futures;
        futures.reserve(numOrganismsDel);

		for (int i = 0; i < settings.numEpoch; i++)
		{
			std::sort(organisms.begin(), organisms.end(), fitnessCmpFn);

			Log( "epoch: ", i);
			Log( "rankings: ");

			organisms.at(0)->DisplayFull();//organisms.at(1), organisms.at(2));

			// don't evolve on the last epoch
			if (i < settings.numEpoch - 1)
			{
				for (int j = numOrganismsSave; j < settings.numPopulation; j++)
				{
                    futures.emplace_back(
                        workers.push( std::bind(
                            EvolveThenEval,
                            organisms.at(j).get(),
                            organisms.at(parentIndexDist()).get(),
                            organisms.at(parentIndexDist()).get(),
                            mutProbDist(),
                            (typename OrganismBase::EvolveType) childCreatorDist() )));
				}
                
                for (auto& it : futures)
                {
                    it.get();
                }
                futures.clear();
			}
		}

        Log( "completed");
        Log( "rankings: ");
		for (auto& it : organisms)
		{
			it->DisplayFull();
		}
	}

	void Run()
	{
		auto mutationRNG = UserRNG::GetRngFn(
			settings.minMutationPercent, settings.maxMutationPercent);

		if (settings.useIntType)
		{
			auto weightRNG = 
				UserRNG::GetRngFn((int) settings.minWeight, (int) settings.maxWeight);

			_Run(mutationRNG, weightRNG);
		}
		else
		{
			auto weightRNG = 
				UserRNG::GetRngFn(settings.minWeight, settings.maxWeight);

			_Run(mutationRNG, weightRNG);
		}
	};

private:
    
    const FitnessFn& fitnessFn;
    const CreateFn& createFn;
    Settings settings;
    ThreadPool workers;
};

#endif

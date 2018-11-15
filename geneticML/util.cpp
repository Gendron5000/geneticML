#include "util.h"

_LogContents::LogString _LogContents::logContents(
    new std::stringstream(), WriteToFileThenDelete() );

void ReportFatalError(const std::string& error)
{
    Log(error);
    std::exit(1);
}

arma::cube GetInputDataExitOnError(const std::string& fileName)
{
	const CubeWError dataVar = GetInputData(fileName);

	if (auto* error = std::get_if<ErrMsg>(&dataVar); error != nullptr)
	{
		ReportFatalError(*error);
	}

	return std::get<arma::cube>(dataVar);
}

CubeWError GetInputData(const std::string& in_fileName)
{
using Json = nlohmann::json;
using StrJsonMap = std::unordered_map<std::string, Json>;

using TickData = std::tuple<float, float, float, float, long>;
using TimeTickMap = std::map<time_t, TickData>;

	TimeTickMap impData;
	Json jsonBase;

	if(std::ifstream file(in_fileName); file.is_open())
	{
		file >> jsonBase;
	}
	else
	{
		return "error: could not open filename: "s + in_fileName;
	}

	try
	{
		StrJsonMap dataBase = jsonBase.get<StrJsonMap>();
		for(auto& it1 : dataBase)
		{
			if (it1.first.substr(0, 11) != "Time Series")
				continue;

			StrJsonMap data2 = it1.second.get<StrJsonMap>();

			for(auto& it2 : data2)
			{
				struct tm timeS;
                std::string timeStr = it2.first;
				Json tickData = it2.second;

				strptime(timeStr.c_str(), "%Y-%m-%d %H:%M:%S", &timeS);

				impData.emplace( mktime(&timeS), TickData(
						std::stof(tickData.at("1. open").get<std::string>()),
						std::stof(tickData.at("2. high").get<std::string>()),
						std::stof(tickData.at("3. low").get<std::string>()),
						std::stof(tickData.at("4. close").get<std::string>()),
						std::stol(tickData.at("5. volume").get<std::string>()) ));
			}
		}
	}
	catch(std::exception const & e)
	{
		return "error: std exception: data might not be in proper format in file: "s + in_fileName;
	}
	catch(...)
	{
		return "error: unknown exception: data might not be in proper format in file: "s + in_fileName;
	}

	if (impData.size() == 0)	
	{
		return "error: could not parse data from file: "s + in_fileName;
	}

	arma::cube inputData = arma::zeros<arma::cube>(1, impData.size(), 1);

	int count = 0;
	double lastVal = std::get<0>((*impData.begin()).second);
	for(auto& it : impData)
	{
		Log(it.first);
		inputData(0,count++,0) = (lastVal - std::get<0>(it.second) );
		lastVal = std::get<0>(it.second);
	}

	return inputData;
}

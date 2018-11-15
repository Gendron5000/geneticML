#ifndef UTIL_H
#define UTIL_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <variant>

#include <mlpack/prereqs.hpp>
#include <nlohmann/json.hpp>

using namespace std::string_literals;

// used for writing an object (ie stringstream) to a file before being deleted
struct WriteToFileThenDelete
{
	template <class T>
	void operator()(T* p)
	{
		if(std::ofstream file("/tmp/log.txt"); file.is_open())
		{
			file << p->str().c_str();
		}
		delete p;
	}
};

struct _LogContents {
using LogString = std::unique_ptr<std::stringstream, WriteToFileThenDelete>;
    static LogString logContents;
};

template <typename... Args>
void LogToStream(std::ostream& stream, Args&&... args)
{
	stream << "\n";
	((stream << args), ...);
}

template <typename... Args>
void Log(Args&&... args)
{
    LogToStream(*_LogContents::logContents, args...);
	LogToStream(std::cout, args...);
}

void ReportFatalError(const std::string& error);

using ErrMsg = std::string;
using CubeWError = std::variant<arma::cube, ErrMsg>;

CubeWError GetInputData(const std::string& in_fileName);
arma::cube GetInputDataExitOnError(const std::string& fileName);

#endif

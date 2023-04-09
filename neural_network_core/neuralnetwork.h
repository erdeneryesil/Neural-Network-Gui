#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include"cell.h"
#include"data.h"
#include<string>
#include<mutex>//for GUI
#include<future>//for GUI

class FormMain;
//-------------------------------------------------
class InputLayerPattern{
protected:
  size_t _countOriginal;
  Data::Type* _types;
  Data::Limit** _limits;
  size_t* _lengths;
  size_t _totalLength;
public:
  InputLayerPattern(const size_t& countOriginal,const Data::Type*const types,const Data::Limit*const*const limits);
  InputLayerPattern(const size_t& countOriginal,const Data::Type& type,const Data::Limit& limit);
  InputLayerPattern(const InputLayerPattern& inputLayerPatternObj);
  ~InputLayerPattern();
  size_t getCountOriginal()const;
  size_t getLength(const size_t& index)const;
  size_t getTotalLength()const;
  Data::Type getType(const size_t& index)const;
  const Data::Limit* getLimit(const size_t& index)const;
  TransformedData** createSample(const int*const originals)const;
};
//-------------------------------------------------
class OutputLayerPattern:public InputLayerPattern{
  double* _tolerances;
  Cell::SummationFunction* _sumFunc;
  Cell::ActivationFunction* _actFunc;
public:
  OutputLayerPattern(const size_t& countOriginal,const Data::Type*const types,const Data::Limit*const*const limits,const double*const tolerances,const Cell::SummationFunction::Type& sumFuncType,const Cell::ActivationFunction::Type& actFuncType);
  OutputLayerPattern(const size_t& countOriginal,const Data::Type& type,const Data::Limit& limit,const double& tolerance,const Cell::SummationFunction::Type& sumFuncType,const Cell::ActivationFunction::Type& actFuncType);
  OutputLayerPattern(const OutputLayerPattern& outputLayerPatternObj);
  ~OutputLayerPattern();
  double getTolerance(const size_t& index)const;
  const Cell::SummationFunction* getSumFunc()const;
  const Cell::ActivationFunction* getActFunc()const;
  void setSumFunc(const Cell::SummationFunction::Type& sumFuncType);
  void setActFunc(const Cell::ActivationFunction::Type& actFuncType);
  void setTolerance(const size_t& index,const double& tolerance);
};
//-------------------------------------------------
class HiddenLayerPattern{
  size_t _countLayer;
  size_t *_countsCell;
  Cell::SummationFunction** _sumFuncs;
  Cell::ActivationFunction** _actFuncs;
public:
  HiddenLayerPattern(const size_t& countLayer,const size_t*const countsCell,const Cell::SummationFunction::Type*const sumFuncTypes,const Cell::ActivationFunction::Type*const actFuncTypes);
  HiddenLayerPattern(const HiddenLayerPattern& hiddenLayerPatternObj);
  ~HiddenLayerPattern();
  size_t getCountLayer()const;
  size_t getCountCell(const size_t& index)const;
  const Cell::SummationFunction* getSumFunc(const size_t& index)const;
  const Cell::ActivationFunction* getActFunc(const size_t& index)const;
  void setSumFunc(const size_t& index,const Cell::SummationFunction::Type& sumFuncType);
  void setActFunc(const size_t& index,const Cell::ActivationFunction::Type& actFuncType);
};
//-------------------------------------------------
class LogEW{
private:
  const HiddenLayerPattern*const _hiddenLayerPatternOfNN;
  const OutputLayerPattern*const _outputLayerPatternOfNN;
  const char*const _separatorOfNN;
  const double*const _lambdaOfNN;
  const double*const _alfaOfNN;

  std::string _file;

  size_t _countTraining;
  Cell::SummationFunction::Type** _sumFuncTypesOfHiddenLayer;
  Cell::ActivationFunction::Type** _actFuncTypesOfHiddenLayer;
  Cell::SummationFunction::Type* _sumFuncTypesOfOutputLayer;
  Cell::ActivationFunction::Type* _actFuncTypesOfOutputLayer;
  double** _tolerances;
  double* _lambdas;
  double* _alfas;
  int* _countsSample;
  size_t* _countsEW;//EW is an error value or an updated weight
  double** _errorValues;
  size_t** _updatedWeights;

  double _newErrorValue;
  size_t _newUpdatedWeight;

  void _loadOneTrainingFromFile(const size_t& indexTraining,const std::string& lineFromFile);
  void _clearData(size_t countLoadedTrainig);

public:
  LogEW(const std::string& fileEW,const HiddenLayerPattern*const hiddenLayerPatternOfNN,const OutputLayerPattern*const outputLayerPatternOfNN,const char*const separatorOfNN,const double*const lambdaOfNN,const double*const alfaOfNN);
  ~LogEW();
  void createNewFile()const;
  std::string getFile()const;
  bool isFileEmpty()const;

  void addNewTrainingInFile(const int& countSample)const;
  void addErrorInFile(const double& errorValue,const size_t& updatedWeight)const;

  void loadTrainingsFromFile();

  size_t getCountTraining()const;
  Cell::SummationFunction::Type getSumFuncTypeOfHiddenLayer(const size_t& indexTraining,const size_t& indexHiddenLayer)const;
  Cell::ActivationFunction::Type getActFuncTypeOfHiddenLayer(const size_t& indexTraining,const size_t& indexHiddenLayer)const;
  Cell::SummationFunction::Type getSumFuncTypeOfOutputLayer(const size_t& indexTraining)const;
  Cell::ActivationFunction::Type getActFuncTypeOfOutputLayer(const size_t& indexTraining)const;
  double getTolerance(const size_t& indexTraining,const size_t& indexTolerance)const;
  double getLambda(const size_t& indexTraining)const;
  double getAlfa(const size_t& indexTraining)const;
  int getCountSample(const size_t& indexTraining)const;
  size_t getCountEWInATraining(const size_t& indexTraining)const;
  double getErrorValue(const size_t& indexTraining,const size_t& indexErrorValue)const;
  size_t getUpdatedWeight(const size_t& indexTraining,const size_t& indexUpdatedWeight)const;

  void setNewErrorValue(const double& errorValue);
  double getNewErrorValue()const;
  void setNewUpdatedWeight(const size_t& updatedWeight);
  size_t getNewUpdatedWeight()const;
};
//-------------------------------------------------
class NeuralNetwork{
public:
    enum class ActionMode{Idle,Train,Test,Run};//for GUI
    enum class StatusTraining{Ongoing,Halted,Succesfull,Failed};//for GUI
    enum class VerificationSamplesFile{None,Verified,Denied};//for GUI

    NeuralNetwork(const double& lambda,const double& alfa,const InputLayerPattern& inputLayerPattern,const HiddenLayerPattern& hiddenLayerPattern,const OutputLayerPattern& outputLayerPattern,const std::string& parametersFile,const char& separator,const double& randomWeightMin,const double& randomWeightMax);
    NeuralNetwork(const std::string& parametersFile,const char& separator);
    ~NeuralNetwork();

    double getLambda()const;
    double getAlfa()const;
    int getEpoch()const;
    char getSeparator()const;
    std::string getParametersFile()const;
    const HiddenLayerPattern* getHiddenLayerPattern()const;
    const OutputLayerPattern* getOutputLayerPattern()const;
    const InputLayerPattern* getInputLayerPattern()const;
    const LogEW* getLogEW()const;

    std::mutex* getMutex();//for GUI
    ActionMode getActionMode()const;//for GUI
    int getCountSample()const;//for GUI
    double getResultTest()const;//for GUI
    int getCountErrorOfTest()const;//for GUI
    int getCountAllOutputOfTest()const;//for GUI
    int getIndexSampleOfTest()const;//for GUI
    size_t getIndexOutputOfTest()const;//for GUI
    double getOutputOfTest()const;//for GUI
    double getExpectedOutputOfTest()const;//for GUI
    double getSingleOutputOfRun()const;//for GUI
    size_t getIndexSingleOutputOfRun()const;//for GUI
    VerificationSamplesFile getVerificationSamplesFile()const;//for GUI
    void resetVerificationSamplesFile();//for GUI
    void setStopTraining(const bool& stop);//for GUI
    StatusTraining getStatusTraining()const;//for GUI
    std::string getTrainingErrorMessage()const;//for GUI
    int getCountLinesOfFile(const std::string& file);

    void setLambda(const double& lambda);
    void setAlfa(const double& alfa);
    void setSumFuncOfHiddenLayer(const size_t& index,const Cell::SummationFunction::Type& sumFuncType);
    void setActFuncOfHiddenLayer(const size_t& index,const Cell::ActivationFunction::Type& actFuncType);
    void setSumFuncOfOutputLayer(const Cell::SummationFunction::Type& sumFuncType);
    void setActFuncOfOutputLayer(const Cell::ActivationFunction::Type& actFuncType);
    void setTolerance(const size_t& index,const double& tolerance);
    void saveNeuralNetwork()const;

    bool train(const int& countSample, const std::string& trainingSamplesFile);
    double* run(const int*const inputsOriginal);
    double test(const int& countSample,const std::string& testSamplesFile);

    void trainForGUI(const int& countSampleCorrected, const std::string& trainingSamplesFile);//for GUI
    void runForGUI(const int*const inputsOriginal,std::promise<bool> &promiseFinish);//for GUI
    void testForGUI(const int& countSampleCorrected,const std::string& testSamplesFile);//for GUI

private:
    InputLayerPattern* _inputLayerPattern;
    InputCell** _inputCells;

    OutputLayerPattern* _outputLayerPattern;
    OutputCell** _outputCells;

    HiddenLayerPattern* _hiddenLayerPattern;
    HiddenCell*** _hiddenCells;

    size_t _countBiasCell;
    BiasCell** _biasCells;

    double _lambda;
    double _alfa;
    int _epoch;

    char _separator;
    std::string _parametersFile;

    LogEW* _logEW;

    VerificationSamplesFile _verificationSamplesFile;//for GUI

    StatusTraining _statusTraining;//for GUI
    int _countSample;//for GUI
    bool _stopTraining;//for GUI
    ActionMode _actionMode;//for GUI
    std::mutex* _mutex;//for GUI

    double _resultTest;//for GUI
    int _countErrorOfTest;//for GUI
    int _countAllOutputOfTest;//for GUI
    int _indexSampleOfTest;//for GUI
    size_t _indexOutputOfTest;//for GUI
    double _outputOfTest;//for GUI
    double _expectedOutputOfTest;//for GUI

    double _singleOutputOfRun;//forGUI
    size_t _indexSingleOutputOfRun;//for GUI

    std::string _trainingErrorMessage;//forGUI


    int _correctCountSampleInFile(const int& countSample, const std::string& samplesFile);
    const int*const* _convertSamplesFileToArray(const int& countSample, const std::string& samplesFile);
    void _calculateNetsOfHiddenLayer(const size_t& indexHiddenLayer);
    void _calculateOutputsOfHiddenLayer(const size_t& indexHiddenLayer);
    void _calculateNetsOfOutputLayer();
    void _calculateOutputsOfOutputLayer();
    void _calculateErrorsOfOutputLayer();
    void _calculateDeltasOfOutputLayer();
    void _calculateDeltasOfHiddenLayer(const size_t& indexHiddenLayer);
    void _calculateDeltaInputWeightsOfOutputLayer();
    void _calculateDeltaInputWeightsOfHiddenLayer(const size_t& indexHiddenLayer);
    void _updateInputWeightsOfOutputLayer();
    void _updateInputWeightsOfHiddenLayer(const size_t& indexHiddenLayer);
    void _deleteTrainingArrays(const int& countSample,const int*const*const trainingSamplesOriginal,const TransformedData*const*const*const trainingSamplesInputsTransformed,const TransformedData*const*const*const trainingSamplesOutputsTransformed);

    bool _controlError(const double& error)const;
    bool _controlWeights()const;
    bool _controlOutputs(const int& countSample, const int*const*const trainingSamplesOriginal);

};
//-------------------------------------------------
#endif // NEURAL_NETWORK_H

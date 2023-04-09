#ifndef CELL_H
#define CELL_H

#include"definescore.h"
#include<cstddef>

//-------------------------------------------------
class Cell
{
public:
  //-------------------------------------------------
  class SummationFunction{
  public:
    enum class Type{WeightedTotal,Multiplication,Maximum,Minimum,Majority,IncrementalTotal};
    typedef double(SummationFunction::*FunctionPointer)(const Cell*const,const double*const)const;
    SummationFunction(const SummationFunction::Type& type);
    Type getType()const;
    FunctionPointer getFunctionPointer()const;
  private:
    Type _type;
    FunctionPointer _functionPointer;
    double _weightedTotal(const Cell*const cell,const double*const inputs)const;
    double _multiplication(const Cell*const cell,const double*const inputs)const;
    double _maximum(const Cell*const cell,const double*const inputs)const;
    double _minimum(const Cell*const cell,const double*const inputs)const;
    double _majority(const Cell*const cell,const double*const inputs)const;
    double _incrementalTotal(const Cell*const cell,const double*const inputs)const;
  };
  //-------------------------------------------------
  class ActivationFunction{
  public:
    enum class Type{Sigmoid,TanH,ReLU,LeakyReLU,Swish,Softplus};
    typedef double(ActivationFunction::*FunctionPointer)(const Cell*const)const;

    ActivationFunction(const ActivationFunction::Type& type);
    static double rangeMin(const ActivationFunction::Type& type);
    static double rangeMax(const ActivationFunction::Type& type);
    Type getType()const;
    FunctionPointer getActivationFunctionPointer()const;
    FunctionPointer getDerivativeActivationFunctionPointer()const;

  private:
    Type _type;
    FunctionPointer _functionPointer;
    FunctionPointer _derivativeFunctionPointer;

    double _sigmoid(const Cell*const cell)const;
    double _tanH(const Cell*const cell)const;
    double _reLU(const Cell*const cell)const;
    double _leakyReLU(const Cell*const cell)const;
    double _swish(const Cell*const cell)const;
    double _softplus(const Cell*const cell)const;

    double _sigmoidDerivative(const Cell*const cell)const;
    double _tanHDerivative(const Cell*const cell)const;
    double _reLUDerivative(const Cell*const cell)const;
    double _leakyReLUDerivative(const Cell*const cell)const;
    double _swishDerivative(const Cell*const cell)const;
    double _softplusDerivative(const Cell*const cell)const;
  };
  //-------------------------------------------------
  Cell(const size_t& countWeight,const double*const inputWeights);
  Cell(const size_t& countWeight,const double& randomWeightMin,const double& randomWeightMax);
  ~Cell();
  size_t getCountWeight()const;
  double getInputWeight(const size_t& index)const;
  double getDeltaInputWeight(const size_t& index)const;
  double getNet()const;
  double getOutput()const;
  void setOutput(const double& output);
  double getDelta()const;
  void calculateNet(const SummationFunction*const summationFunction,const double*const inputs);
  void calculateOutput(const ActivationFunction*const activationFunction);
  void calculateDeltaInputWeight(const double& lambda,const double& alfa, const double*const previousLayerOutputs);
  void updateInputWeight();
private:
  void _setWeightRandom(const double& min,const double& max);
protected:
  size_t _countWeight;
  double* _inputWeights;
  double* _deltaInputWeights;
  double _net;
  double _output;
  double _delta;
};
//-------------------------------------------------
class BiasCell
{
protected:
  double _input;
public:
  BiasCell();
  double getInput()const;
};
//-------------------------------------------------
class OutputCell:public Cell
{
private:
  double _expectedOutput;
  double _error;
public:
  OutputCell(const size_t& countWeight,const double*const inputWeights);
  OutputCell(const size_t& countWeight,const double& randomWeightMin,const double& randomWeightMax);

  double getExpectedOutput()const;
  void setExpectedOutput(const double& value);
  double getError()const;
  void setError(const double& error);
  void calculateError();
  void calculateDelta(const ActivationFunction*const activationFunction);
};
//-------------------------------------------------
class HiddenCell:public Cell
{
public:
  HiddenCell(const size_t& countWeight,const double*const inputWeights);
  HiddenCell(const size_t& countWeight,const double& randomWeightMin,const double& randomWeightMax);
  void calculateDelta(const ActivationFunction*const activationFunction,const int& cellIndex,const int& countPreviousLayerCell,const Cell*const*const previousLayerCells);
};
//-------------------------------------------------
class InputCell:public BiasCell
{
public:
  InputCell();
  void setInput(const double& input);
};
//-------------------------------------------------
#endif // CELL_H

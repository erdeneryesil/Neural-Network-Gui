#ifndef DATA_H
#define DATA_H

#include"definescore.h"
#include<math.h>
#include<stdexcept>

enum class LayerType{Input,Hidden,Output};
class Data{
protected:
  double*_digits;
  size_t _length;
public:
  virtual ~Data();
  size_t getLength()const;
  virtual double getValue(const double&,const double&)const{throw std::invalid_argument(THROW_MESSAGE_DATA_GETVALUE_2_PARAMETERS);};
  virtual double getValue(const size_t&,const double&,const double&)const{throw std::invalid_argument(THROW_MESSAGE_DATA_GETVALUE_3_PARAMETERS);}
  virtual double getOriginal()const{throw std::invalid_argument(THROW_MESSAGE_DATA_GETORIGINAL_NO_PARAMETER);}
  virtual double getOriginal(const int&,const int&)const{throw std::invalid_argument(THROW_MESSAGE_DATA_GETORIGINAL_2_PARAMETERS);};
  enum class Type{Binary,Flag,Narrowed,Native};
  class Limit{
  private:
    int _originalMin;
    int _originalMax;
    double _scaledMin;
    double _scaledMax;
  public:
    Limit();
    Limit(const Data::Type& dataType,const LayerType& layerType);//Input - Native
    Limit(const Data::Type& dataType,const LayerType& layerType,const double& scaledMin,const double& scaledMax);//Output - Native
    Limit(const Data::Type& dataType,const LayerType& layerType,const int& originalMin,const int& originalMax);//Input - Binary,Flag,Narrowed
    Limit(const Data::Type& dataType,const LayerType& layerType,const int& originalMin,const int& originalMax,const double& scaledMin,const double& scaledMax);//Output - Binary,Flag,Narrowed
    Limit(const Limit*const limit);
    int getOriginalMin()const;
    int getOriginalMax()const;
    double getScaledMin()const;
    double getScaledMax()const;
    void setScaledMin(const LayerType& layerType,const double& scaledMin);//only Output layer
    void setScaledMax(const LayerType& layerType,const double& scaledMax);//only Output layer
  };
};
//-------------------------------------------------
class Binary:public Data{
private:
    void _calculateDigitsReverseOrder(int original);
    void _reverseBits();
public:
    Binary(const int& original,const int& originalMax);
    Binary(const size_t& countBit,const double*const scaledBits,const int& originalMax,const double& scaledMin,const double& scaledMax);
    Binary(const Binary& binaryObj);
    double getValue(const size_t& index,const double& scaledMin,const double& scaledMax)const;
    double getOriginal()const;
};
//-------------------------------------------------
//toDecimalValue() fonksiyonu en büyük(1'e en yakın) flag verisine göre sonuç döndürür. Aynı büyük değerden 2 tane varsa soldan ilkine göre işlem yapar
class Flag:public Data{
public:
  Flag(const int& original,const int& originalMax);
  Flag(const size_t& countFlag,const double*const scaledFlags,const int& originalMax,const double& scaledMin,const double& scaledMax);
  Flag(const Flag& flagObj);
  double getValue(const size_t& index,const double& scaledMin,const double& scaledMax)const;
  double getOriginal()const;
};
//-------------------------------------------------
class Narrowed:public Data{
public:
  Narrowed(const double& value,const double& originalMin,const double& originalMax);
  Narrowed(const Narrowed& narrowedObj);
  double getValue(const double& scaledMin,const double& scaledMax)const;
  double getOriginal(const int& originalMin,const int& originalMax)const;
};
//-------------------------------------------------
class Native:public Data{
public:
  Native(const double& value,const double& originalMin,const double& originalMax);//if it is an output data
  Native(const double& native);
  Native(const Native& nativeObj);
  double getValue(const double& scaledMin,const double& scaledMax)const;
  double getOriginal()const;
};
//-------------------------------------------------
class TransformedData{
public:
  static size_t length(const Data::Type& type,const int& original);
  TransformedData(Binary* data);
  TransformedData(Flag* data);
  TransformedData(Narrowed* data);
  TransformedData(Native* data);
  TransformedData(const TransformedData& data);
  ~TransformedData();
  Data::Type getType()const;
  const Data* getData()const;
private:
  Data* _data;
  Data::Type _type;
};
//-------------------------------------------------
#endif // DATA_H

#include "neuron.h"
#include <cmath>


double neuron::eta = 0.15; // net learning rate
double neuron::alpha = 0.5; // momentum

 neuron::neuron(unsigned numOutputs, unsigned myIndex)
{
	for (unsigned c = 0; c < numOutputs; ++c) {
		outputWeights.push_back(Connection());
		outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
} //random weights to weights in the connection structure

void neuron::updateInputWeights(Layer &prevLayer)
{
	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight = eta * neuron.getOutputVal() * gradient + alpha * oldDeltaWeight;
		neuron.outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}   //update the Input Weights by updating weights and deltaweights

double neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += outputWeights[n].weight * nextLayer[n].gradient;
	}

	return sum;
} //sum by multipliyng weights and gradient

void neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	gradient = dow * neuron::activationFunctionDerivative(outputVal);
} //find gradient by formula
void neuron::calcOutputGradients(double targetVals)
{
	double delta = targetVals - outputVal;
	gradient = delta * neuron::activationFunctionDerivative(outputVal);
}   //find gradient by formula

double neuron::activationFunction(double x)
{
	//output range [-1.0..1.0]
	return tanh(x);
}

double neuron::activationFunctionDerivative(double x)
{
	return 1.0 - x * x;
}

void neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;

	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputVal() *
			prevLayer[n].outputWeights[m_myIndex].weight;
	}      //sum of the all all neuron output and weights in the layer

	outputVal = neuron::activationFunction(sum);
}

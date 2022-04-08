package network

import (
	"fmt"
	"math"

	"github.com/AlexandreMarcq/nn/matrix"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

func generateWeights(size int, v float64) (weights []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	weights = make([]float64, size)
	for i := 0; i < size; i++ {
		weights[i] = dist.Rand()
	}

	return
}

func sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}

func sigmoidPrime(m mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	o := make([]float64, r)
	for i := 0; i < r; i++ {
		o[i] = 1
	}
	ones := mat.NewDense(r, 1, o)
	return matrix.Multiply(m, matrix.Substract(ones, m))
}

type Network struct {
	InputLayersNumber  int
	hiddenLayersNumber int
	OutputLayerNumber  int
	HiddenWeights      *mat.Dense
	OutputWeights      *mat.Dense
	learningRate       float64
}

func NewNetwork(inputs, hiddens, outputs int, rate float64) (net *Network) {
	net = &Network{
		InputLayersNumber:  inputs,
		hiddenLayersNumber: hiddens,
		OutputLayerNumber:  outputs,
		learningRate:       rate,
	}

	net.HiddenWeights = mat.NewDense(
		net.hiddenLayersNumber,
		inputs,
		generateWeights(net.InputLayersNumber*net.hiddenLayersNumber, float64(net.InputLayersNumber)),
	)

	net.OutputWeights = mat.NewDense(
		outputs,
		net.hiddenLayersNumber,
		generateWeights(net.hiddenLayersNumber*net.OutputLayerNumber, float64(net.hiddenLayersNumber)),
	)

	return
}

func (n Network) Predict(inputData []float64) (mat.Matrix, error) {
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs, err := matrix.Dot(n.HiddenWeights, inputs)
	if err != nil {
		return nil, err
	}
	hiddenOutputs := matrix.Apply(sigmoid, hiddenInputs)
	finalInputs, err := matrix.Dot(n.OutputWeights, hiddenOutputs)
	if err != nil {
		return nil, err
	}
	outputData := matrix.Apply(sigmoid, finalInputs)
	return outputData, nil
}

func (n *Network) Train(inputData []float64, targetData []float64) error {
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs, err := matrix.Dot(n.HiddenWeights, inputs)
	if err != nil {
		return err
	}
	hiddenOutputs := matrix.Apply(sigmoid, hiddenInputs)
	finalInputs, err := matrix.Dot(n.OutputWeights, hiddenOutputs)
	if err != nil {
		return err
	}
	finalOutputs := matrix.Apply(sigmoid, finalInputs)

	targets := mat.NewDense(len(targetData), 1, targetData)
	outputErrors := matrix.Substract(targets, finalOutputs)
	hiddenErrors, err := matrix.Dot(n.OutputWeights.T(), outputErrors)
	if err != nil {
		return err
	}

	// delta_wjk = - l * (t_k - o_k) . o_k * (1 - o_k) . o_j
	//           = - learning_rate * outputErrors . finalOutputs * (1 - finalOutputs) . hiddenOutputs

	a := matrix.Multiply(outputErrors, sigmoidPrime(finalOutputs))
	b, err := matrix.Dot(a, hiddenOutputs.T())
	if err != nil {
		return err
	}
	c := matrix.Scale(b, n.learningRate)
	n.OutputWeights = matrix.Add(n.OutputWeights, c).(*mat.Dense)

	d := matrix.Multiply(hiddenErrors, sigmoidPrime(hiddenOutputs))
	e, err := matrix.Dot(d, hiddenOutputs.T())
	if err != nil {
		return err
	}
	f := matrix.Scale(e, n.learningRate)
	n.HiddenWeights = matrix.Add(n.HiddenWeights, f).(*mat.Dense)

	return nil
}

func (n Network) String() string {
	return fmt.Sprintf(
		"Number of input nodes: %d\nNumber of hidden layers: %d\nNumber of output nodes: %d\nLearning rate: %f\n",
		n.InputLayersNumber,
		n.hiddenLayersNumber,
		n.OutputLayerNumber,
		n.learningRate,
	)
}

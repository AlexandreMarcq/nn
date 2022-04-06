package mnist

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/AlexandreMarcq/nn/network"
	"github.com/AlexandreMarcq/nn/utils"
)

func Train(net *network.Network, epochs int) error {
	rand.Seed(time.Now().UTC().UnixNano())

	t1 := time.Now()

	f, err := os.Open("data/mnist_train.csv")
	if err != nil {
		return err
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return err
	}

	size := len(records)

	for e := 0; e < epochs; e++ {
		epoch_fmt := fmt.Sprintf("\x0cEpoch %d/%d\n", e+1, epochs)
		for i, record := range records {
			progress := float64(i) / float64(size) * 100.0
			bar := int(math.Floor(progress))
			fmt.Printf(epoch_fmt+"Record %d/%d\n[%-100s]  -  %.2f%%\n", i+1, size, strings.Repeat("=", bar)+">", progress)

			inputs := make([]float64, net.InputLayersNumber)
			for j := range inputs {
				x, err := strconv.ParseFloat(record[j], 64)
				if err != nil {
					return err
				}
				inputs[j] = (x / 255.0 * 0.99) + 0.01
			}

			targets := make([]float64, 10)
			for j := range targets {
				targets[j] = 0.01
			}
			x, err := strconv.Atoi(record[0])
			if err != nil {
				return err
			}
			targets[x] = 0.99

			err = net.Train(inputs, targets)
			if err != nil {
				return err
			}
		}
	}

	elapsed := time.Since(t1)
	fmt.Printf("Done training, took %s\n", elapsed)
	return nil
}

func Predict(net *network.Network) error {
	t1 := time.Now()

	f, err := os.Open("data/mnist_test.csv")
	if err != nil {
		return err
	}
	defer f.Close()

	score := 0
	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return err
	}

	size := len(records)

	for i, record := range records {
		progress := float64(i) / float64(size) * 100.0
		bar := int(math.Floor(progress))
		fmt.Printf("Record %d/%d\n[%-100s]  -  %.2f%%\n", i+1, size, strings.Repeat("=", bar)+">", progress)

		inputs := make([]float64, net.InputLayersNumber)
		for j := range inputs {
			x, err := strconv.ParseFloat(record[j], 64)
			if err != nil {
				return err
			}
			inputs[j] = (x / 255.0 * 0.99) + 0.01
		}
		outputs, err := net.Predict(inputs)
		if err != nil {
			return err
		}
		best := 0
		highest := 0.0
		for j := 0; j < net.OutputLayerNumber; j++ {
			if outputs.At(j, 0) > highest {
				best = j
				highest = outputs.At(j, 0)
			}
		}
		target, err := strconv.Atoi(record[0])
		if best == target {
			score++
		}
	}

	elapsed := time.Since(t1)
	fmt.Printf("Done, checking took: %s\n", elapsed)
	fmt.Printf("Score: %.2f%%\n", float64(score)/float64(size)*100.0)

	return nil
}

func PredictFromImage(net *network.Network, path string) (int, []float64, error) {
	input, err := utils.DataFromImage(path)
	if err != nil {
		return -1, nil, err
	}
	output, err := net.Predict(input)
	if err != nil {
		return -1, nil, err
	}
	best := 0
	highest := 0.0
	predictions := make([]float64, net.OutputLayerNumber)
	for i := 0; i < net.OutputLayerNumber; i++ {
		if output.At(i, 0) > highest {
			best = i
			highest = output.At(i, 0)
		}
		predictions[i] = output.At(i, 0)
	}
	return best, predictions, nil
}

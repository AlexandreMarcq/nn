package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/AlexandreMarcq/nn/mnist"
	"github.com/AlexandreMarcq/nn/network"
	"github.com/AlexandreMarcq/nn/utils"
)

func main() {
	action := flag.String("action", "", "Train, predict a batch or predict a single value: train|predict_batch|predict")
	//dataType := flag.String("type", "", "Type of data: mnist")

	learningRate := flag.Float64("rate", 0.1, "Learning rate of the network: defaults to 0.1")

	flag.Parse()

	net := network.NewNetwork(784, 10, 3, *learningRate)

	switch *action {
	case "train":
		err := mnist.Train(net, 5)
		if err != nil {
			fmt.Println("Error while training:", err)
		}
		err = utils.Save(net, "./saves")
		if err != nil {
			fmt.Println("Error while saving network:", err)
		}
	case "predict_batch":
		utils.Load(net, "./saves")
		mnist.Predict(net)
	case "predict":
		utils.Load(net, "./saves")
		num, outputs, err := mnist.PredictFromImage(net, "./data/nums/0.png")
		if err != nil {
			fmt.Println("Error:", err)
		}
		fmt.Println("Prediction:", num)
		fmt.Println("Outputs:")
		for i, output := range outputs {
			fmt.Printf("%d => %05.2f %%\n", i, output*100)
		}
	default:
		fmt.Println("Expected 'train', 'predict_batch', or 'predict' for -action flag.")
		os.Exit(1)
	}
}

package utils

import (
	"fmt"
	"image"
	"image/png"
	"os"

	"github.com/AlexandreMarcq/nn/network"
)

func DataFromImage(path string) ([]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	img, err := png.Decode(f)
	if err != nil {
		return nil, err
	}

	bounds := img.Bounds()
	gray := image.NewGray(bounds)

	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {
			rgba := img.At(x, y)
			gray.Set(x, y, rgba)
		}
	}

	pixels := make([]float64, len(gray.Pix))

	for i := 0; i < len(gray.Pix); i++ {
		pixels[i] = (float64(255-gray.Pix[i]) / 255.0 * 0.99) + 0.01
	}

	return pixels, nil
}

func Load(net *network.Network, path string) error {
	for i, hiddenLayer := range net.HiddenWeights {
		h, err := os.Open(path + fmt.Sprintf("/test/hweights%d.model", i))
		defer h.Close()
		if err != nil {
			return err
		}
		hiddenLayer.Reset()
		hiddenLayer.UnmarshalBinaryFrom(h)
	}

	o, err := os.Open(path + "/test/oweights.model")
	defer o.Close()
	if err != nil {
		return err
	}
	net.OutputWeights.Reset()
	net.OutputWeights.UnmarshalBinaryFrom(o)

	return nil
}

func Save(net *network.Network, path string) error {
	for i, hiddenLayer := range net.HiddenWeights {
		h, err := os.Create(path + fmt.Sprintf("/test/hweights%d.model", i))
		defer h.Close()
		if err != nil {
			return err
		}
		hiddenLayer.MarshalBinaryTo(h)
	}

	o, err := os.Create(path + "/test/oweights.model")
	defer o.Close()
	if err != nil {
		return err
	}

	net.OutputWeights.MarshalBinaryTo(o)

	return nil
}

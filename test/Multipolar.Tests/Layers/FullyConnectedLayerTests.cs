using Multipolar.Layers;
using Xunit;

namespace Multipolar.Tests.Layers
{
    public class FullyConnectedLayerTests
    {
        [Fact]
        public void KitchenSink()
        {
            var layer = new FullyConnectedLayer(3, 2);

            // Node 0

            layer.Weights[0] = 1;
            layer.Weights[2] = 2;
            layer.Weights[4] = 3;

            layer.Biases[0] = 1;

            // Node 1

            layer.Weights[1] = 4;
            layer.Weights[3] = 5;
            layer.Weights[5] = 6;

            layer.Biases[1] = 2;

            // Feed

            var input = new float[3];

            input[0] = 1;
            input[1] = 2;
            input[2] = 3;

            layer.Feed(input);

            Assert.Equal((1 * 1 + 2 * 2 + 3 * 3) + 1, layer.Output[0]);
            Assert.Equal((4 * 1 + 5 * 2 + 6 * 3) + 2, layer.Output[1]);

            // ComputeGradient

            var gradient = new float[2];

            gradient[0] = 1;
            gradient[1] = 2;

            layer.ComputeGradient(gradient);

            Assert.Equal(9, layer.InputGradient[0]);
            Assert.Equal(12, layer.InputGradient[1]);
            Assert.Equal(15, layer.InputGradient[2]);

            // Optimize

            layer.Optimize(input, 0.5f);

            Assert.Equal(0.5f, layer.Weights[0]);
            Assert.Equal(1.0f, layer.Weights[2]);
            Assert.Equal(1.5f, layer.Weights[4]);
            Assert.Equal(0.5f, layer.Biases[0]);

            Assert.Equal(3.0f, layer.Weights[1]);
            Assert.Equal(3.0f, layer.Weights[3]);
            Assert.Equal(3.0f, layer.Weights[5]);
            Assert.Equal(1.0f, layer.Biases[1]);
        }
    }
}

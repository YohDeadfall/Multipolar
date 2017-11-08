using Multipolar.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Xunit;

namespace Multipolar.Tests.Layers
{
    public class TwoByTwoPoolingLayerTests
    {
        [Fact]
        public void KitchenSink()
        {
            var layer = new TwoByTwoPoolingLayer((2, 2, 2));

            var input = new float[2, 4, 4]
            {
                {
                    { 1, 2, 3, 4 },
                    { 8, 7, 6, 5 },
                    { 1, 2, 4, 3 },
                    { 7, 8, 6, 5 },
                },
                {
                    { 4, 3, 2, 1 },
                    { 5, 6, 7, 8 },
                    { 3, 4, 2, 1 },
                    { 5, 6, 8, 7 },
                },
            }
            .Transpose(1, 2, 0);

            // Feed

            layer.Feed(input);

            var expectedOutput = new float[2, 2, 2]
            {
                {
                    { 8, 6 },
                    { 8, 6 },
                },
                {
                    { 6, 8 },
                    { 6, 8 },
                }
            }
            .Transpose(1, 2, 0);

            Assert.Equal(expectedOutput.Cast<float>(), layer.Output.Cast<float>());

            var expectedSelection = new float[2, 4, 4]
            {
                {
                    { 0, 0, 0, 0 },
                    { 1, 0, 1, 0 },
                    { 0, 0, 0, 0 },
                    { 0, 1, 1, 0 },
                },
                {
                    { 0, 0, 0, 0 },
                    { 0, 1, 0, 1 },
                    { 0, 0, 0, 0 },
                    { 0, 1, 1, 0 },
                },
            }
            .Transpose(1, 2, 0);

            Assert.Equal(expectedSelection.Cast<float>(), layer.Selection.Cast<float>());

            // ComputeGradient

            layer.ComputeGradient(new float[8] { 2, 3, 4, 6, 7, 8, 0, 1 });

            var expectedGradient = new float[2, 4, 4]
            {
                {
                    { 00, 00, 00, 00 },
                    { 24, 00, 48, 00 },
                    { 00, 00, 00, 00 },
                    { 00, 00, 08, 00 },
                },
                {
                    { 00, 00, 00, 00 },
                    { 00, 24, 00, 48 },
                    { 00, 00, 00, 00 },
                    { 00, 00, 08, 00 },
                },
            }
            .Transpose(1, 2, 0);

            Assert.Equal(expectedGradient.Cast<float>(), layer.InputGradient.Cast<float>());
        }
    }
}

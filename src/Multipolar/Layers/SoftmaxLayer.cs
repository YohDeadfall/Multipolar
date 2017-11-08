using System;
using static System.Diagnostics.Debug;

namespace Multipolar.Layers
{
    public unsafe class SoftmaxLayer
    {
        public readonly int Size;

        public readonly float[] Output;

        public readonly float[] InputGradient;

        public SoftmaxLayer(int size)
        {
            Size = size;

            Output = new float[size];
            InputGradient = new float[size];
        }

        public void Feed(float[] input)
        {
            Assert(input.Length == Size);

            fixed (float* p_input = &input[0])
            {
                Feed(p_input);
            }
        }

        public void ComputeGradient(float[] previous)
        {
            Assert(previous.Length == Size);

            fixed (float* p_previous = &previous[0])
            {
                ComputeGradient(p_previous);
            }
        }

        private void Feed(float* p_input)
        {
            fixed (float* p_output = &Output[0])
            {
                var max = float.MinValue;
                var scale = 0f;

                for (var i = 0; i < Size; i++)
                {
                    max = Math.Max(max, p_input[i]);
                }

                for (var i = 0; i < Size; i++)
                {
                    scale += p_output[i] = (float)Math.Exp(p_input[i] - max);
                }

                for (var i = 0; i < Size; i++)
                {
                    p_output[i] /= scale;
                }
            }
        }

        private void ComputeGradient(float* p_previous)
        {
            for (var i = 0; i < InputGradient.Length; i++)
            {
                InputGradient[i] = p_previous[i];
            }
        }
    }
}

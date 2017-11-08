using System;
using static System.Diagnostics.Debug;

namespace Multipolar.Layers
{
    public unsafe class DropoutLayer
    {
        private readonly Random random = new Random();

        public readonly int Size;

        public readonly float Probability;

        public readonly float[] Keeps;

        public readonly float[] Output;

        public readonly float[] InputGradient;

        public DropoutLayer(int size, float probability)
        {
            Size = size;
            Probability = probability;
            Keeps = new float[size];
            Output = new float[size];
            InputGradient = new float[size];

            Array.Fill(Keeps, 1, 0, (int)Math.Ceiling(size * probability));
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

            for (var i = 0; i < Size; i++)
            {
                InputGradient[i] = previous[i] * Keeps[i];
            }
        }

        private void Feed(float* p_input)
        {
            for (var i = 0; i < Size - 1; i++)
            {
                var j = random.Next(i, Size);
                var k = Keeps[j];

                Keeps[j] = Keeps[i];
                Keeps[i] = k;

                Output[i] = p_input[i] * k;
            }
        }
    }
}

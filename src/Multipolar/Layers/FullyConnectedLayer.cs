using System.Numerics;
using System.Runtime.CompilerServices;
using static System.Diagnostics.Debug;

namespace Multipolar.Layers
{
    public unsafe class FullyConnectedLayer
    {
        public readonly int Inputs;

        public readonly int Outputs;

        public readonly float[] Weights;

        public readonly float[] Biases;

        public readonly float[] Output;

        public readonly float[] InputGradient;

        public readonly float[] OutputGradient;

        public FullyConnectedLayer(int inputs, int outputs)
        {
            Inputs = inputs;
            Outputs = outputs;
            Weights = new float[inputs * outputs];
            Biases = new float[outputs];
            Output = new float[outputs];
            InputGradient = new float[inputs];
            OutputGradient = new float[outputs];
        }

        public void Feed(float[] input)
        {
            Assert(input.Length == Inputs);

            fixed (float* p_input = &input[0])
            {
                Feed(p_input);
            }
        }

        public void Feed(float[,,] input)
        {
            Assert(input.Length == Inputs);

            fixed (float* p_input = &input[0, 0, 0])
            {
                Feed(p_input);
            }
        }

        public void ComputeGradient(float[] previous)
        {
            Assert(previous.Length == Outputs);

            fixed (float* p_previous = &previous[0])
            {
                ComputeGradient(p_previous);
            }
        }

        public void Optimize(float[] input, float eta)
        {
            Assert(input.Length == Inputs);

            fixed (float* p_input = &input[0])
            {
                Optimize(p_input, eta);
            }
        }

        public void Optimize(float[,,] input, float eta)
        {
            Assert(input.Length == Inputs);

            fixed (float* p_input = &input[0, 0, 0])
            {
                Optimize(p_input, eta);
            }
        }

        private void Feed(float* p_input)
        {
            var n_vectors = Outputs / Vector<float>.Count;
            var r_vectors = Outputs % Vector<float>.Count;

            fixed (float* p_output = &Output[0])
            fixed (float* p_weights = &Weights[0])
            fixed (float* p_biases = &Biases[0])
            {
                for (var i_output = 0; i_output < Outputs; i_output++)
                {
                    p_output[i_output] = p_biases[i_output];
                }

                var i_weights = 0;

                for (var i_input = 0; i_input < Inputs; i_input++)
                {
                    var i_output = 0;
                    var v_input = new Vector<float>(p_input[i_input]);

                    for (var i_vector = 0; i_vector < n_vectors; i_vector++)
                    {
                        var v_weights = Unsafe.Read<Vector<float>>(p_weights + i_weights);
                        var v_output = Unsafe.Read<Vector<float>>(p_output + i_output);

                        (v_output + v_input * v_weights).CopyTo(Output, i_output);

                        i_weights += Vector<float>.Count;
                        i_output += Vector<float>.Count;
                    }

                    for (var i_vector = 0; i_vector < r_vectors; i_vector++)
                    {
                        p_output[i_output] += p_input[i_input] * p_weights[i_weights];

                        i_weights++;
                        i_output++;
                    }
                }
            }
        }

        private void ComputeGradient(float* p_previous)
        {
            var n_vectors = Outputs / Vector<float>.Count;
            var r_vectors = Outputs % Vector<float>.Count;

            fixed (float* p_output_g = &OutputGradient[0])
            fixed (float* p_weights = &Weights[0])
            fixed (float* p_input_g = &InputGradient[0])
            {
                for (var i_output = 0; i_output < Outputs; i_output++)
                {
                    p_output_g[i_output] = p_previous[i_output];
                }

                var i_weights = 0;

                for (var i_input = 0; i_input < Inputs; i_input++)
                {
                    var input_g = 0f;
                    var i_output = 0;

                    for (var i_vector = 0; i_vector < n_vectors; i_vector++)
                    {
                        var v_weights = Unsafe.Read<Vector<float>>(p_weights + i_weights);
                        var v_output_g = Unsafe.Read<Vector<float>>(p_output_g + i_output);

                        input_g += Vector.Dot(v_output_g, v_weights);

                        i_weights += Vector<float>.Count;
                        i_output += Vector<float>.Count;
                    }

                    for (var i_vector = 0; i_vector < r_vectors; i_vector++)
                    {
                        input_g += p_output_g[i_output] * p_weights[i_weights];

                        i_weights++;
                        i_output++;
                    }

                    p_input_g[i_input] = input_g;
                }
            }
        }

        private void Optimize(float* p_input, float eta)
        {
            var n_vectors = Outputs / Vector<float>.Count;
            var r_vectors = Outputs % Vector<float>.Count;

            fixed (float* p_output_g = &OutputGradient[0])
            fixed (float* p_weights = &Weights[0])
            fixed (float* p_biases = &Biases[0])
            {
                for (var i_output = 0; i_output < Outputs; i_output++)
                {
                    p_biases[i_output] -= p_output_g[i_output] * eta;
                }

                var i_weights = 0;

                for (var i_input = 0; i_input < Inputs; i_input++)
                {
                    var i_output = 0;
                    var e_input = p_input[i_input] * eta;
                    var v_input = new Vector<float>(e_input);

                    for (var i_vector = 0; i_vector < n_vectors; i_vector++)
                    {
                        var v_weights = Unsafe.Read<Vector<float>>(p_weights + i_weights);
                        var v_output_g = Unsafe.Read<Vector<float>>(p_output_g + i_output);
                        var v_change = v_output_g * v_input;

                        (v_weights - v_change).CopyTo(Weights, i_weights);

                        i_weights += Vector<float>.Count;
                        i_output += Vector<float>.Count;
                    }

                    for (var i_vector = 0; i_vector < r_vectors; i_vector++)
                    {
                        var change = p_output_g[i_output] * e_input;

                        p_weights[i_weights] -= change;

                        i_weights++;
                        i_output++;
                    }
                }
            }
        }
    }
}

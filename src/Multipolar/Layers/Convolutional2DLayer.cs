using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using static System.Diagnostics.Debug;

namespace Multipolar.Layers
{
    public unsafe class Convolutional2DLayer
    {
        // Using readonly fields because properties + ValueTuples cause a measurable
        // performance degradation, but the tuples are so much nicer to work with than
        // the alternative.

        public readonly (int Height, int Width, int Channels) InputDimensions;

        public readonly (int Height, int Width, int Channels) OutputDimensions;

        public readonly (int Height, int Width) KernelDimensions;

        public readonly (int X, int Y) Stride;

        public readonly (int Top, int Right, int Bottom, int Left) Padding;

        public readonly float[] Kernel;

        public readonly float[] Biases;

        public readonly float[] Output;

        public readonly float[] InputGradient;

        public readonly float[] OutputGradient;

        public Convolutional2DLayer(
            int inputHeight,
            int inputWidth,
            int inputChannels,
            int kernelHeight,
            int kernelWidth,
            int outputChannels)
        {
            InputDimensions = (inputHeight, inputWidth, inputChannels);

            KernelDimensions = (kernelHeight, kernelWidth);

            Kernel = new float[kernelHeight * kernelWidth * inputChannels * outputChannels];

            Biases = new float[outputChannels];

            Output = new float[inputHeight * inputWidth * outputChannels];

            InputGradient = new float[inputHeight * inputWidth * inputChannels];

            // For now let's assume a stride of 1/1 and padding 'same'.

            Stride = (1, 1);

            var paddingY = ((kernelHeight - 1) / 2);
            var paddingX = ((kernelWidth - 1) / 2);

            Padding = (paddingY, paddingX, paddingY, paddingX);

            var outputHeight = ((Padding.Top + InputDimensions.Height + Padding.Bottom) - kernelHeight) / Stride.Y + 1;
            var outputWidth = ((Padding.Left + InputDimensions.Width + Padding.Right) - kernelWidth) / Stride.X + 1;

            OutputDimensions = (outputHeight, outputWidth, outputChannels);

            OutputGradient = new float[outputHeight * outputWidth * outputChannels];
        }

        public void Feed(float[] input)
        {
            Assert(input.Length == InputGradient.Length);

            fixed (float* p_input = &input[0])
            {
                Feed(p_input);
            }
        }

        public void Feed(float[,,] input)
        {
            Assert(input.Length == InputGradient.Length);

            fixed (float* p_input = &input[0, 0, 0])
            {
                Feed(p_input);
            }
        }

        public void ComputeGradient(float[] previous)
        {
            Assert(previous.Length == Output.Length);

            fixed (float* p_previous = &previous[0])
            {
                ComputeGradient(p_previous);
            }
        }

        public void Optimize(float[] input, float eta)
        {
            Assert(input.Length == InputGradient.Length);

            fixed (float* p_input = &input[0])
            {
                Optimize(p_input, eta);
            }
        }

        public void Optimize(float[,,] input, float eta)
        {
            Assert(input.Length == InputGradient.Length);

            fixed (float* p_input = &input[0, 0, 0])
            {
                Optimize(p_input, eta);
            }
        }

        private void Feed(float* p_input)
        {
            var n_vectors = InputDimensions.Channels / Vector<float>.Count;
            var r_vectors = InputDimensions.Channels % Vector<float>.Count;

            fixed (float* p_output = &Output[0])
            fixed (float* p_kernel = &Kernel[0])
            fixed (float* p_biases = &Biases[0])
            {
                var i_output = 0;

                for (var output_y = 0; output_y < OutputDimensions.Height; output_y++)
                {
                    for (var output_x = 0; output_x < OutputDimensions.Width; output_x++)
                    {
                        var (input_y_bounds, input_x_bounds) = GetPatchBounds(output_y, output_x);

                        for (var output_c = 0; output_c < OutputDimensions.Channels; output_c++)
                        {
                            p_output[i_output + output_c] = p_biases[output_c];
                        }

                        for (var input_y = input_y_bounds.start; input_y < input_y_bounds.end; input_y++)
                        {
                            for (var input_x = input_x_bounds.start; input_x < input_x_bounds.end; input_x++)
                            {
                                var kernel_y = input_y - input_y_bounds.start;
                                var kernel_x = input_x - input_x_bounds.start;
                                var i_kernel = GetKernelIndex(kernel_y, kernel_x);
                                var i_input = GetInputIndex(input_y, input_x);

                                for (var output_c = 0; output_c < OutputDimensions.Channels; output_c++)
                                {
                                    var j_kernel = i_kernel;
                                    var j_input = i_input;
                                    var sum = 0f;

                                    for (var i_vector = 0; i_vector < n_vectors; i_vector++)
                                    {
                                        var v_input = Unsafe.Read<Vector<float>>(p_input + j_input);
                                        var v_kernel = Unsafe.Read<Vector<float>>(p_kernel + j_kernel);

                                        sum += Vector.Dot(v_input, v_kernel);

                                        j_kernel += Vector<float>.Count;
                                        j_input += Vector<float>.Count;
                                    }

                                    for (var i_vector = 0; i_vector < r_vectors; i_vector++)
                                    {
                                        sum += p_input[j_input] * p_kernel[j_kernel];

                                        j_kernel++;
                                        j_input++;
                                    }

                                    // It is measurably faster to sum up in the local variable and 
                                    // assign to the referenced location in one go.
                                    p_output[i_output + output_c] += sum;
                                }
                            }
                        }

                        i_output += OutputDimensions.Channels;
                    }
                }
            }
        }

        private void ComputeGradient(float* p_previous)
        {
            var n_vectors = InputDimensions.Channels / Vector<float>.Count;
            var r_vectors = InputDimensions.Channels % Vector<float>.Count;

            fixed (float* p_input_g = &InputGradient[0])
            fixed (float* p_output_g = &OutputGradient[0])
            fixed (float* p_kernel = &Kernel[0])
            {
                for (var i_input = 0; i_input < InputGradient.Length; i_input++)
                {
                    p_input_g[i_input] = 0f;
                }

                var i_output = 0;

                for (var output_y = 0; output_y < OutputDimensions.Height; output_y++)
                {
                    for (var output_x = 0; output_x < OutputDimensions.Width; output_x++)
                    {
                        var (input_y_bounds, input_x_bounds) = GetPatchBounds(output_y, output_x);

                        for (var output_c = 0; output_c < OutputDimensions.Channels; output_c++)
                        {
                            p_output_g[i_output + output_c] = p_previous[i_output + output_c];
                        }

                        for (var input_y = input_y_bounds.start; input_y < input_y_bounds.end; input_y++)
                        {
                            for (var input_x = input_x_bounds.start; input_x < input_x_bounds.end; input_x++)
                            {
                                var kernel_y = input_y_bounds.end - input_y - 1;
                                var kernel_x = input_x_bounds.end - input_x - 1;
                                var i_kernel = GetKernelIndex(kernel_y, kernel_x);
                                var i_input = GetInputIndex(input_y, input_x);

                                for (var output_c = 0; output_c < OutputDimensions.Channels; output_c++)
                                {
                                    var j_kernel = i_kernel;
                                    var j_input = i_input;
                                    var output_g = p_output_g[i_output + output_c];
                                    var v_output = new Vector<float>(output_g);

                                    for (var i_vector = 0; i_vector < n_vectors; i_vector++)
                                    {
                                        var v_input = Unsafe.Read<Vector<float>>(p_input_g + j_input);
                                        var v_kernel = Unsafe.Read<Vector<float>>(p_kernel + j_kernel);
                                        
                                        (v_input + v_output * v_kernel).CopyTo(InputGradient, j_input);

                                        j_kernel += Vector<float>.Count;
                                        j_input += Vector<float>.Count;
                                    }

                                    for (var i_vector = 0; i_vector < r_vectors; i_vector++)
                                    {
                                        p_input_g[j_input] += output_g * p_kernel[j_kernel];

                                        j_kernel++;
                                        j_input++;
                                    }
                                }
                            }
                        }

                        i_output += OutputDimensions.Channels;
                    }
                }
            }
        }

        private void Optimize(float* p_input, float eta)
        {
            var n_vectors = InputDimensions.Channels / Vector<float>.Count;
            var r_vectors = InputDimensions.Channels % Vector<float>.Count;

            fixed (float* p_output_g = &OutputGradient[0])
            fixed (float* p_kernel = &Kernel[0])
            fixed (float* p_biases = &Biases[0])
            {
                var i_output = 0;

                for (var output_y = 0; output_y < OutputDimensions.Height; output_y++)
                {
                    for (var output_x = 0; output_x < OutputDimensions.Width; output_x++)
                    {
                        var (input_y_bounds, input_x_bounds) = GetPatchBounds(output_y, output_x);

                        for (var output_c = 0; output_c < OutputDimensions.Channels; output_c++)
                        {
                            p_biases[output_c] -= p_output_g[i_output + output_c] * eta;
                        }

                        for (var input_y = input_y_bounds.start; input_y < input_y_bounds.end; input_y++)
                        {
                            for (var input_x = input_x_bounds.start; input_x < input_x_bounds.end; input_x++)
                            {
                                var kernel_y = input_y_bounds.end - input_y - 1;
                                var kernel_x = input_x_bounds.end - input_x - 1;
                                var i_kernel = GetKernelIndex(kernel_y, kernel_x);
                                var i_input = GetInputIndex(input_y, input_x);

                                for (var output_c = 0; output_c < OutputDimensions.Channels; output_c++)
                                {
                                    var j_kernel = i_kernel;
                                    var j_input = i_input;
                                    var e_output_g = p_output_g[i_output + output_c] * eta;
                                    var v_output = new Vector<float>(e_output_g);

                                    for (var i_vector = 0; i_vector < n_vectors; i_vector++)
                                    {
                                        var v_kernel = Unsafe.Read<Vector<float>>(p_kernel + j_kernel);
                                        var v_input = Unsafe.Read<Vector<float>>(p_input + j_input);

                                        (v_kernel - v_input * v_output).CopyTo(Kernel, j_kernel);

                                        j_kernel += Vector<float>.Count;
                                        j_input += Vector<float>.Count;
                                    }

                                    for (var i_vector = 0; i_vector < r_vectors; i_vector++)
                                    {
                                        p_kernel[j_kernel] -= p_input[j_input] * e_output_g;

                                        j_kernel++;
                                        j_input++;
                                    }
                                }
                            }
                        }

                        i_output += OutputDimensions.Channels;
                    }
                }
            }
        }

        private ((int start, int end), (int start, int end)) GetPatchBounds(int output_y, int output_x)
        {
            var input_y_base = output_y * Stride.Y - Padding.Top;
            var input_y_start = Math.Max(0, input_y_base);
            var input_y_end = Math.Min(InputDimensions.Height, input_y_base + KernelDimensions.Height);

            var input_x_base = output_x * Stride.X - Padding.Left;
            var input_x_start = Math.Max(0, input_x_base);
            var input_x_end = Math.Min(InputDimensions.Width, input_x_base + KernelDimensions.Width);

            return ((input_y_start, input_y_end), (input_x_start, input_x_end));
        }

        private int GetInputIndex(int input_y, int input_x)
        {
            var offset_y = input_y * InputDimensions.Width * InputDimensions.Channels;
            var offset_x = input_x * InputDimensions.Channels;

            return offset_y + offset_x;
        }

        private int GetKernelIndex(int kernel_y, int kernel_x)
        {
            var offset_y = kernel_y * KernelDimensions.Width * OutputDimensions.Channels * InputDimensions.Channels;
            var offset_x = kernel_x * OutputDimensions.Channels * InputDimensions.Channels;

            return offset_y + offset_x;
        }
    }
}

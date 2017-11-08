using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Multipolar
{
    public static class Extensions
    {
        // Thanks to https://stackoverflow.com/a/1136335/1030925
        public static int MaxIndex<T>(this IEnumerable<T> source)
        {
            IComparer<T> comparer = Comparer<T>.Default;
            using (var iterator = source.GetEnumerator())
            {
                if (!iterator.MoveNext())
                {
                    throw new InvalidOperationException("Empty sequence");
                }
                int maxIndex = 0;
                T maxElement = iterator.Current;
                int index = 0;
                while (iterator.MoveNext())
                {
                    index++;
                    T element = iterator.Current;
                    if (comparer.Compare(element, maxElement) > 0)
                    {
                        maxElement = element;
                        maxIndex = index;
                    }
                }
                return maxIndex;
            }
        }

        public static unsafe void Fill<T>(this Array array, IEnumerable<T> values) where T : struct
        {
            if (array.GetType().GetElementType() != typeof(T))
            {
                throw new InvalidOperationException("Types do not match");
            }

            var handle = GCHandle.Alloc(array, GCHandleType.Pinned);

            try
            {
                var pointer = handle.AddrOfPinnedObject().ToPointer();

                using (var enumerator = values.GetEnumerator())
                {
                    int i;

                    for (i = 0; i < array.Length && enumerator.MoveNext(); i++)
                    {
                        Unsafe.Add(ref Unsafe.AsRef<T>(pointer), i) = enumerator.Current;
                    }

                    Debug.Assert(i == array.Length);
                }
            }
            finally
            {
                handle.Free();
            }
        }

        private static unsafe bool MatchRank(int rank, int[] dimensions)
        {
            if (rank != dimensions.Length)
            {
                return false;
            }

            var found = stackalloc bool[rank];

            for (var i = 0; i < rank; i++)
            {
                var dimension = dimensions[i];

                if (found[dimension])
                {
                    return false;
                }

                found[dimension] = true;
            }

            return true;
        }

        public static unsafe TArray Transpose<TArray>(this TArray array, params int[] dimensions)
        {
            if (!array.GetType().IsArray)
            {
                throw new InvalidOperationException("Must be an array type");
            }

            var typed = (Array)(object)array;

            if (!MatchRank(typed.Rank, dimensions))
            {
                throw new InvalidOperationException("The provided dimensions are not compatible with the provided array");
            }

            return (TArray)TransposeMethodInfo
                .MakeGenericMethod(array.GetType().GetElementType())
                .Invoke(null, new object[] { array, dimensions });
        }

        private static readonly MethodInfo TransposeMethodInfo
            = typeof(Extensions).GetMethod(nameof(Transpose), BindingFlags.NonPublic | BindingFlags.Static);

        private static unsafe object Transpose<TElement>(this Array array, int[] dimensions)
        {
            GCHandle handle_in = default, handle_out = default;

            try
            {
                var in_sizes = new int[array.Rank];
                var in_mods = new int[array.Rank];

                for (var i = 0; i < array.Rank; i++)
                {
                    in_sizes[i] = 1;

                    for (var j = i + 1; j < array.Rank; j++)
                    {
                        in_sizes[i] *= array.GetLength(j);
                    }
                }

                in_mods[0] = array.Length;

                for (var i = 1; i < array.Rank; i++)
                {
                    in_mods[i] = in_sizes[i - 1];
                }

                var out_lengths = new int[array.Rank];
                var out_sizes = new int[array.Rank];

                for (var out_dim = 0; out_dim < dimensions.Length; out_dim++)
                {
                    var in_dim = dimensions[out_dim];

                    out_lengths[out_dim] = array.GetLength(in_dim);

                    out_sizes[in_dim] = 1;

                    for (var j = out_dim + 1; j < array.Rank; j++)
                    {
                        out_sizes[in_dim] *= array.GetLength(dimensions[j]);
                    }
                }

                var result = Array.CreateInstance(typeof(TElement), out_lengths);

                handle_in = GCHandle.Alloc(array, GCHandleType.Pinned);
                handle_out = GCHandle.Alloc(result, GCHandleType.Pinned);

                var pointer_in = handle_in.AddrOfPinnedObject().ToPointer();
                var pointer_out = handle_out.AddrOfPinnedObject().ToPointer();

                for (var i_in = 0; i_in < array.Length; i_in++)
                {
                    var i_out = 0;

                    for (var d = 0; d < array.Rank; d++)
                    {
                        i_out += ((i_in % in_mods[d]) / in_sizes[d]) * out_sizes[d];
                    }

                    Unsafe.Add(ref Unsafe.AsRef<TElement>(pointer_out), i_out) = Unsafe.Add(ref Unsafe.AsRef<TElement>(pointer_in), i_in);
                }

                return result;
            }
            finally
            {
                if (handle_in.IsAllocated)
                {
                    handle_in.Free();
                }

                if (handle_out.IsAllocated)
                {
                    handle_out.Free();
                }
            }
        }
    }
}

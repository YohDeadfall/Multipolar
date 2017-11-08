using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace Multipolar
{
    public static class IDXFileReader
    {
        public async static Task<Array> Load(string filename)
        {
            using (var file = File.OpenRead(filename))
            {
                var buffer = new byte[4096];

                await file.ReadAsync(buffer, 0, 4);

                var elementType = buffer[2];
                var dimensions = buffer[3];
                var lengths = new int[dimensions];

                await file.ReadAsync(buffer, 0, 4 * dimensions);

                for (var i = 0; i < dimensions; i++)
                {
                    lengths[i] = ReadBigEndianInt32(buffer, 4 * i);
                }

                switch (elementType)
                {
                    case 0x08:
                        return await Load<byte>(file, buffer, lengths);

                    case 0x09:
                        return await Load<sbyte>(file, buffer, lengths);

                    case 0x0B:
                        return await Load<short>(file, buffer, lengths);

                    case 0x0C:
                        return await Load<int>(file, buffer, lengths);

                    case 0x0D:
                        return await Load<float>(file, buffer, lengths);

                    case 0x0E:
                        return await Load<double>(file, buffer, lengths);

                    default:
                        throw new NotSupportedException();
                }
            }
        }

        private static async Task<Array> Load<TElement>(Stream stream, byte[] buffer, int[] lengths)
        {
            var array = Array.CreateInstance(typeof(TElement), lengths);
            var handle = GCHandle.Alloc(array, GCHandleType.Pinned);

            try
            {
                var element = 0;
                var size = Unsafe.SizeOf<TElement>();

                ReadChunk:

                var read = await stream.ReadAsync(buffer, 0, buffer.Length);

                if (read > 0)
                {
                    unsafe
                    {
                        var pointer = handle.AddrOfPinnedObject().ToPointer();

                        for (var offset = 0; offset < read; offset += size)
                        {
                            TElement value = default;

                            if (typeof(TElement) == typeof(byte))
                            {
                                value = (TElement)(object)buffer[offset];
                            }
                            else if (typeof(TElement) == typeof(sbyte))
                            {
                                value = (TElement)(object)(sbyte)buffer[offset];
                            }
                            else if (typeof(TElement) == typeof(short))
                            {
                                value = (TElement)(object)ReadBigEndianShort(buffer, offset);
                            }
                            else if (typeof(TElement) == typeof(int))
                            {
                                value = (TElement)(object)ReadBigEndianInt32(buffer, offset);
                            }
                            else if (typeof(TElement) == typeof(float))
                            {
                                value = (TElement)(object)BitConverter.ToSingle(buffer, offset);
                            }
                            else if (typeof(TElement) == typeof(double))
                            {
                                value = (TElement)(object)BitConverter.ToDouble(buffer, offset);
                            }

                            Unsafe.Add(ref Unsafe.AsRef<TElement>(pointer), element) = value;

                            element++;
                        }
                    }

                    goto ReadChunk;
                }

                return array;
            }
            finally
            {
                handle.Free();
            }
        }

        private static int ReadBigEndianInt32(byte[] buffer, int offset)
        {
            var result = 0;

            result |= buffer[offset + 0] << 24;
            result |= buffer[offset + 1] << 16;
            result |= buffer[offset + 2] << 08;
            result |= buffer[offset + 3] << 00;

            return result;
        }

        private static short ReadBigEndianShort(byte[] buffer, int offset)
        {
            var result = 0;
            
            result |= buffer[offset + 0] << 08;
            result |= buffer[offset + 1] << 00;

            return (short)result;
        }
    }
}

const math = @import("math.zig");
const std = @import("std");

pub fn DenseLayer(comptime batch_size: comptime_int, comptime inputs: comptime_int, comptime neurons: comptime_int) type {
    return struct {
        weights: math.Mat(inputs, neurons),
        biases: @Vector(neurons, f64),
        output: math.Mat(batch_size, neurons),

        const Self = @This();
        pub const Inputs = inputs;
        pub const Neurons = neurons;
        pub const BatchSize = batch_size;

        pub fn init() Self {
            var prng = std.Random.DefaultPrng.init(0); // Fixed seed for reproducibility or use std.crypto.random.int(u64)
            const random = prng.random();

            var randomSlice: [inputs * neurons]f64 = undefined;
            for (0..inputs * neurons) |i| {
                randomSlice[i] = 0.01 * random.floatNorm(f64);
            }

            const weights = math.Mat(inputs, neurons).init(&randomSlice);
            const biases: @Vector(neurons, f64) = @splat(0);

            // Initialize output with zeros
            const output_zeroes = [_]f64{0} ** (batch_size * neurons);
            const output = math.Mat(batch_size, neurons).init(&output_zeroes);

            return Self{ .weights = weights, .biases = biases, .output = output };
        }

        pub fn forward(self: *Self, input: math.Mat(batch_size, inputs)) void {
            // X * W
            var partial = input.dot(neurons, self.weights);

            // Add biases to every row
            for (0..batch_size) |r| {
                partial.values[r] += self.biases;
            }

            self.output = partial;
        }
    };
}

pub fn ActivationRelu(comptime batch_size: comptime_int, comptime neurons: comptime_int) type {
    return struct {
        output: math.Mat(batch_size, neurons),

        const Self = @This();
        pub const inputs = neurons;
        pub const Inputs = inputs;
        pub const Neurons = neurons;
        pub const BatchSize = batch_size;

        pub fn init() Self {
            const output_zeroes = [_]f64{0} ** (batch_size * neurons);
            const output = math.Mat(batch_size, neurons).init(&output_zeroes);

            return Self{ .output = output };
        }

        pub fn forward(self: *Self, input: math.Mat(batch_size, neurons)) void {
            const zero_vec: @Vector(neurons, f64) = @splat(0.0);
            for (input.values, 0..) |row, i| {
                self.output.values[i] = @max(row, zero_vec);
            }
        }
    };
}
pub fn SoftMaxLayer(comptime batch_size: comptime_int, comptime neurons: comptime_int) type {
    return struct {
        output: math.Mat(batch_size, neurons),

        const Self = @This();
        pub const inputs = neurons;
        pub const Inputs = inputs;
        pub const Neurons = neurons;
        pub const BatchSize = batch_size;

        pub fn init() Self {
            const output_zeroes = [_]f64{0} ** (batch_size * neurons);
            const output = math.Mat(batch_size, neurons).init(&output_zeroes);

            return Self{ .output = output };
        }

        pub fn forward(self: *Self, input: math.Mat(batch_size, neurons)) void {
            for (input.values, 0..) |row, i| {
                const rowmax = @reduce(.Max, row);
                const exp = @exp(row - @as(@Vector(neurons, f64), @splat(rowmax)));
                const exp_sum = @reduce(.Add, exp);
                self.output.values[i] = exp / @as(@Vector(neurons, f64), @splat(exp_sum));
            }
        }
    };
}
test "DenseLayer" {
    const batch_size = 2;
    const inputs = 3;
    const neurons = 2;

    const Layer = DenseLayer(batch_size, inputs, neurons);
    var layer = Layer.init();

    // Create dummy input: 2 samples, 3 features
    // [[1, 2, 3], [4, 5, 6]]
    const input_data = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const input = math.Mat(batch_size, inputs).init(&input_data);

    layer.forward(input);

    // Just check dimensions and that it ran without crashing
    try std.testing.expectEqual(batch_size, Layer.BatchSize);
}

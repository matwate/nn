const math = @import("math.zig");
const std = @import("std");

pub fn DenseLayer(comptime batch_size: comptime_int, comptime inputs: comptime_int, comptime neurons: comptime_int) type {
    return struct {
        weights: math.Mat(inputs, neurons),
        biases: @Vector(neurons, f64),

        output: math.Mat(batch_size, neurons),
        inputs: *const math.Mat(batch_size, inputs),

        dweights: math.Mat(inputs, neurons),
        dinputs: math.Mat(batch_size, inputs),
        dbiases: @Vector(neurons, f64),
        const Self = @This();

        pub fn init() Self {
            var prng = std.Random.DefaultPrng.init(0); // Fixed seed for reproducibility or use std.crypto.random.int(u64)
            const random = prng.random();

            var randomSlice: [inputs * neurons]f64 = undefined;
            for (0..inputs * neurons) |i| {
                randomSlice[i] = 0.1 * random.floatNorm(f64);
            }

            const weights = math.Mat(inputs, neurons).init(&randomSlice);
            const biases: @Vector(neurons, f64) = @splat(0);

            // Initialize output with zeros
            const output_zeroes = [_]f64{0} ** (batch_size * neurons);
            const output = math.Mat(batch_size, neurons).init(&output_zeroes);

            const dinputs = math.Mat(batch_size, inputs).init(&output_zeroes);
            const dweights = math.Mat(inputs, neurons).init(&output_zeroes);
            const dbiases: @Vector(neurons, f64) = @splat(0.0);
            const inputs_placeholder = math.Mat(batch_size, inputs).init(&output_zeroes);

            return Self{ .weights = weights, .biases = biases, .inputs = &inputs_placeholder, .output = output, .dinputs = dinputs, .dweights = dweights, .dbiases = dbiases };
        }

        pub fn forward(self: *Self, input: *const math.Mat(batch_size, inputs)) void {
            self.inputs = input;
            // X * W
            var partial = input.dot(neurons, self.weights);

            // Add biases to every row
            for (0..batch_size) |r| {
                partial.values[r] += self.biases;
            }

            self.output = partial;
        }

        pub fn backward(self: *Self, gradient: math.Mat(batch_size, neurons)) void {
            self.dweights = self.inputs.T().dot(neurons, gradient);
            self.dbiases = @as(@Vector(neurons, f64), @splat(0.0));
            for (0..batch_size) |r| {
                self.dbiases += gradient.values[r];
            }
            self.dinputs = gradient.dot(inputs, self.weights.T());
        }

        pub fn update(self: *Self, learning_rate: f64) void {
            const lr_splat: @Vector(neurons, f64) = @splat(learning_rate);

            for (0..inputs) |idx| {
                self.weights.values[idx] = self.weights.values[idx] - lr_splat * self.dweights.values[idx];
            }

            self.biases = self.biases - lr_splat * self.dbiases;
        }
    };
}

pub fn ActivationRelu(comptime batch_size: comptime_int, comptime neurons: comptime_int) type {
    return struct {
        output: math.Mat(batch_size, neurons),
        inputs: *const math.Mat(batch_size, neurons),

        dinputs: math.Mat(batch_size, neurons),
        const Self = @This();

        pub fn init() Self {
            const output_zeroes = [_]f64{0} ** (batch_size * neurons);
            const output = math.Mat(batch_size, neurons).init(&output_zeroes);

            const dinputs = math.Mat(batch_size, neurons).init(&output_zeroes);
            const inputs = math.Mat(batch_size, neurons).init(&output_zeroes);

            return Self{
                .output = output,
                .inputs = &inputs,
                .dinputs = dinputs,
            };
        }

        pub fn forward(self: *Self, input: *const math.Mat(batch_size, neurons)) void {
            self.inputs = input;
            const zero_vec: @Vector(neurons, f64) = @splat(0.0);
            for (input.values, 0..) |row, i| {
                self.output.values[i] = @max(row, zero_vec);
            }
        }

        pub fn backward(self: *Self, gradient: math.Mat(batch_size, neurons)) void {
            const zero_vec: @Vector(neurons, f64) = @splat(0.0);

            for (0..batch_size) |batchIdx| {
                const input_row = self.inputs.values[batchIdx];
                const grad_row = gradient.values[batchIdx];
                const mask = input_row > zero_vec;
                self.dinputs.values[batchIdx] = @select(f64, mask, grad_row, zero_vec);
            }
        }
    };
}
pub fn ActivationSoftMax(comptime batch_size: comptime_int, comptime neurons: comptime_int) type {
    return struct {
        output: math.Mat(batch_size, neurons),
        dinputs: math.Mat(batch_size, neurons), // Store gradient for next layer

        const Self = @This();

        pub fn init() Self {
            const output_zeroes = [_]f64{0} ** (batch_size * neurons);
            const output = math.Mat(batch_size, neurons).init(&output_zeroes);
            const dinputs_zeroes = [_]f64{0} ** (batch_size * neurons);
            const dinputs = math.Mat(batch_size, neurons).init(&dinputs_zeroes);

            return Self{ .output = output, .dinputs = dinputs };
        }

        pub fn forward(self: *Self, input: math.Mat(batch_size, neurons)) void {
            for (input.values, 0..) |row, i| {
                const rowmax = @reduce(.Max, row);
                const exp = @exp(row - @as(@Vector(neurons, f64), @splat(rowmax)));
                const exp_sum = @reduce(.Add, exp);
                self.output.values[i] = exp / @as(@Vector(neurons, f64), @splat(exp_sum));
            }
        }

        pub fn backward(self: *Self, gradient: math.Mat(batch_size, neurons)) void {
            // We assume 'self.output' contains the probabilities (y)
            // We assume 'gradient' contains d_values (incoming gradient)

            for (0..batch_size) |i| {
                const y = self.output.values[i]; // Vector of probabilities
                const dvals = gradient.values[i]; // Incoming gradient vector

                // 1. Element-wise multiplication: y * dvals
                const s1 = y * dvals;

                // 2. Sum the result into a scalar (dot product logic)
                // This represents the "sum of all gradients"
                const sum = @reduce(.Add, s1);

                // 3. Calculate: dinputs = s1 - (sum * y)
                // We splat the scalar 'sum' to match the vector width of 'y'
                self.dinputs.values[i] = s1 - (y * @as(@Vector(neurons, f64), @splat(sum)));
            }
        }
    };
}
pub fn Activation_Softmax_Loss_CategoricalCrossentropy(comptime batch_size: comptime_int, comptime classes: comptime_int) type {
    return struct {
        output: math.Mat(batch_size, classes), // Probabilities
        dinputs: math.Mat(batch_size, classes), // Gradient for logits
        loss: f64,

        const Self = @This();

        pub fn init() Self {
            const zeroes_out = [_]f64{0} ** (batch_size * classes);
            const output = math.Mat(batch_size, classes).init(&zeroes_out);
            const dinputs = math.Mat(batch_size, classes).init(&zeroes_out);
            return Self{ .output = output, .dinputs = dinputs, .loss = 0.0 };
        }

        pub fn forward(self: *Self, inputs: math.Mat(batch_size, classes), y_true: [batch_size]usize) void {
            // 1. Softmax Forward
            for (inputs.values, 0..) |row, i| {
                const rowmax = @reduce(.Max, row);
                const exp = @exp(row - @as(@Vector(classes, f64), @splat(rowmax)));
                const exp_sum = @reduce(.Add, exp);
                self.output.values[i] = exp / @as(@Vector(classes, f64), @splat(exp_sum));
            }

            // 2. Cross Entropy Loss
            var batch_loss: f64 = 0.0;
            const epsilon = 1e-7;
            for (0..batch_size) |i| {
                const pred_prob = self.output.values[i][y_true[i]];
                const clipped_pred = @min(@max(pred_prob, epsilon), 1.0 - epsilon);
                batch_loss += -@log(clipped_pred);
            }
            self.loss = batch_loss / @as(f64, @floatFromInt(batch_size));
        }

        pub fn backward(self: *Self, y_true: [batch_size]usize) void {
            // Gradient of Combined Softmax + CrossEntropy is (y_pred - y_true) / batch_size
            for (0..batch_size) |i| {
                var grad = self.output.values[i];
                grad[y_true[i]] -= 1.0;
                self.dinputs.values[i] = grad / @as(@Vector(classes, f64), @splat(@as(f64, @floatFromInt(batch_size))));
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

pub fn Loss_CategoricalCrossEntropy(comptime batch_size: comptime_int, comptime classes: comptime_int) type {
    return struct {
        output: f64,

        dinputs: math.Mat(batch_size, classes),

        const Self = @This();
        pub const BatchSize = batch_size;
        pub const Classes = classes;

        pub fn init() Self {
            const zeroes: [batch_size * classes]f64 = @splat(0);
            const dinputs = math.Mat(batch_size, classes).init(&zeroes);
            return Self{ .output = 0.0, .dinputs = dinputs };
        }

        pub fn forward(self: *Self, y_pred: math.Mat(batch_size, classes), y_true: [batch_size]usize) void {
            const epsilon = 1e-7;

            var batch_loss: f64 = 0.0;

            for (0..batch_size) |i| {
                const pred_row = y_pred.values[i];
                const target_class = y_true[i];

                // Ensure target class is within bounds
                std.debug.assert(target_class < classes);

                const pred_prob = pred_row[target_class];

                // Clip to avoid log(0)
                const clipped_pred = @min(@max(pred_prob, epsilon), 1.0 - epsilon);

                // Calculate sample loss: -log(prediction_at_target_class)
                const sample_loss = -@log(clipped_pred);

                batch_loss += sample_loss;
            }

            self.output = batch_loss / @as(f64, @floatFromInt(batch_size));
        }

        pub fn backward(self: *Self, gradient: math.Mat(batch_size, classes), y_true: [batch_size]usize) void {
            for (0..batch_size) |idx| {
                self.dinputs.values[idx] = (-math.oneHot(classes, y_true[idx]) / gradient.values[idx]) / @as(@Vector(classes, f64), @splat(@as(f64, @floatFromInt(batch_size))));
            }
        }
    };
}

pub fn Accuracy(comptime batch_size: comptime_int, comptime classes: comptime_int) type {
    return struct {
        output: f64,
        const Self = @This();
        pub const BatchSize = batch_size;
        pub const Classes = classes;

        pub fn init() Self {
            return Self{ .output = 0.0 };
        }

        pub fn forward(self: *Self, y_pred: math.Mat(batch_size, classes), y_true: [batch_size]usize) void {
            var correct: f64 = 0;
            for (0..batch_size) |i| {
                const pred_row = y_pred.values[i];
                const target_class = y_true[i];

                var arg_max: usize = 0;
                // Argmax the pred_row
                const row_max = @reduce(.Max, pred_row);
                inline for (0..classes) |k| {
                    if (pred_row[k] == row_max) {
                        arg_max = k;
                    }
                }
                if (arg_max == target_class) {
                    correct += 1;
                }
            }
            self.output = correct / @as(f64, batch_size);
        }
    };
}

test "Loss_CategoricalCrossEntropy" {
    const batch_size = 3;
    const classes = 3;
    const LossLayer = Loss_CategoricalCrossEntropy(batch_size, classes);
    var loss_layer = LossLayer.init();

    // 3 samples, 3 classes
    // Sample 1: Target class 0, Pred [0.7, 0.2, 0.1] -> loss -log(0.7) ≈ 0.3566
    // Sample 2: Target class 1, Pred [0.1, 0.8, 0.1] -> loss -log(0.8) ≈ 0.2231
    // Sample 3: Target class 2, Pred [0.2, 0.2, 0.6] -> loss -log(0.6) ≈ 0.5108
    // Mean ≈ (0.3566 + 0.2231 + 0.5108) / 3 ≈ 0.3635

    const y_pred_data = [_]f64{
        0.7, 0.2, 0.1,
        0.1, 0.8, 0.1,
        0.2, 0.2, 0.6,
    };
    // Sparse labels: indices of the correct class
    const y_true = [3]usize{ 0, 1, 2 };

    const y_pred = math.Mat(batch_size, classes).init(&y_pred_data);

    loss_layer.forward(y_pred, y_true);

    try std.testing.expect(loss_layer.output > 0.36);
    try std.testing.expect(loss_layer.output < 0.37);
}
pub fn SGDOptimizer(comptime learning_rate: f64) type {
    return struct {
        const Self = @This();

        pub fn step(comptime Layer: type, layer: *Layer) void {
            layer.update(learning_rate);
        }
    };
}

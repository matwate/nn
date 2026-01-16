const std = @import("std");
const math = @import("math.zig");
const nn = @import("nn.zig");
const spiral = @import("spiral.zig");

pub fn main() !void {
    const huh = spiral.createData(100, 3);
    var dense = nn.DenseLayer(300, 2, 64).init();
    var activation = nn.ActivationRelu(300, 64).init();
    var dense2 = nn.DenseLayer(300, 64, 3).init();
    var loss_activation = nn.Activation_Softmax_Loss_CategoricalCrossentropy(300, 3).init();
    var acc_function = nn.Accuracy(300, 3).init();
    const optimizer = nn.SGDOptimizer(1);

    for (0..10000) |epoch| {
        dense.forward(&huh.points);
        activation.forward(&dense.output);
        dense2.forward(&activation.output);

        loss_activation.forward(dense2.output, huh.labels);

        acc_function.forward(loss_activation.output, huh.labels);

        loss_activation.backward(huh.labels);
        dense2.backward(loss_activation.dinputs);
        activation.backward(dense2.dinputs);
        dense.backward(activation.dinputs);

        optimizer.step(@TypeOf(dense2), &dense2);
        optimizer.step(@TypeOf(dense), &dense);

        if (epoch % 10 == 0) {
            std.debug.print("epoch: {d}, acc: {d}, loss: {d}\n", .{ epoch, acc_function.output, loss_activation.loss });
        }
    }

    const memory_bytes = @sizeOf(@TypeOf(huh)) + @sizeOf(@TypeOf(dense)) + @sizeOf(@TypeOf(activation)) + @sizeOf(@TypeOf(dense2)) + @sizeOf(@TypeOf(loss_activation)) + @sizeOf(optimizer);
    const memory_kb = @as(f64, @floatFromInt(memory_bytes)) / 1024.0;
    const memory_perc = (memory_kb * 100) / (16 * 1024);
    std.debug.print("\nStack Memory Used: {d:.2} KB, {d:.2}% of what is available in your computer \n", .{ memory_kb, memory_perc });
}

test "Practice" {
    _ = @import("spiral.zig");
    _ = @import("nn.zig");
}

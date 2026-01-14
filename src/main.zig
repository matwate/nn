const std = @import("std");
const math = @import("math.zig");
const nn = @import("nn.zig");
const spiral = @import("spiral.zig");

pub fn main() !void {
    const huh = spiral.createData(100, 3);
    var dense = nn.DenseLayer(300, 2, 3).init();
    var activation = nn.ActivationRelu(300, 3).init();
    var dense2 = nn.DenseLayer(300, 3, 3).init();
    var softmax = nn.SoftMaxLayer(300, 3).init();
    dense.forward(huh.points);
    activation.forward(dense.output);
    dense2.forward(dense.output);
    softmax.forward(dense2.output);

    for (0..100) |i| {
        std.debug.print("{any} pred:{any}\n", .{ huh.points.values[i], softmax.output.values[i] });
    }

    const memory_bytes = @sizeOf(@TypeOf(huh)) + @sizeOf(@TypeOf(dense)) + @sizeOf(@TypeOf(activation)) + @sizeOf(@TypeOf(dense2)) + @sizeOf(@TypeOf(softmax));
    const memory_kb = @as(f64, @floatFromInt(memory_bytes)) / 1024.0;
    const memory_perc = (memory_kb * 100) / (16 * 1024);
    std.debug.print("\nStack Memory Used: {d:.2} KB, {d:.2}% of what is available in your computer \n", .{ memory_kb, memory_perc });
}

test {
    _ = @import("spiral.zig");
    _ = @import("nn.zig");
}

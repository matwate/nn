const std = @import("std");
const math = @import("math.zig");

pub fn createData(comptime samples: usize, comptime classes: usize) struct { points: math.Mat(samples * classes, 2), labels: [samples * classes]usize } {
    var X = math.Mat(samples * classes, 2){ .values = undefined };
    var y: [samples * classes]usize = undefined;

    var prng = std.Random.DefaultPrng.init(0);
    const random = prng.random();

    for (0..classes) |class_number| {
        const start_idx = samples * class_number;

        for (0..samples) |i| {
            const idx = start_idx + i;

            // r = np.linspace(0.0, 1, samples)
            const r = if (samples > 1) @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(samples - 1)) else 0.0;

            // t = np.linspace(class_number*4, (class_number+1)*4, samples) + randn*0.2
            const t_start = @as(f64, @floatFromInt(class_number)) * 4.0;
            const t_end = @as(f64, @floatFromInt(class_number + 1)) * 4.0;
            const t_step = if (samples > 1) (t_end - t_start) * (@as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(samples - 1))) else 0.0;

            const noise = random.floatNorm(f64) * 0.2;
            const t = t_start + t_step + noise;

            // X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
            const x1 = r * std.math.sin(t * 2.5);
            const x2 = r * std.math.cos(t * 2.5);

            X.values[idx] = .{ x1, x2 };
            y[idx] = @intCast(class_number);
        }
    }

    return .{ .points = X, .labels = y };
}

test "createData" {
    const data = createData(100, 3);
    try std.testing.expectEqual(300, data.points.values.len);
    try std.testing.expectEqual(300, data.labels.len);
    try std.testing.expectEqual(0, data.labels[0]);
    try std.testing.expectEqual(2, data.labels[299]);
}

const std = @import("std");

pub fn dot(comptime len: comptime_int, a: @Vector(len, f64), b: @Vector(len, f64)) f64 {
    return @reduce(.Add, a * b);
}

pub fn oneHot(comptime len: comptime_int, idx: usize) @Vector(len, f64) {
    std.debug.assert(idx < len);
    var s: @Vector(len, f64) = @splat(0.0);
    s[idx] = 1;
    return s;
}

pub fn Mat(comptime rows: comptime_int, comptime columns: comptime_int) type {
    return struct {
        values: [rows]@Vector(columns, f64),

        const Self = @This();
        pub const Rows = rows;
        pub const Columns = columns;

        pub fn init(s: []const f64) Self {
            std.debug.assert(s.len >= rows * columns);
            var result: Self = undefined;

            var i: usize = 0;
            for (0..rows) |r| {
                var row_vec: @Vector(columns, f64) = undefined;
                inline for (0..columns) |c| {
                    row_vec[c] = s[i];
                    i += 1;
                }
                result.values[r] = row_vec;
            }
            return result;
        }

        pub fn dotV(self: *const Self, v: @Vector(columns, f64)) @Vector(rows, f64) {
            var result: @Vector(rows, f64) = @splat(0.0);

            for (0..rows) |r| {
                result[r] = @reduce(.Add, self.values[r] * v);
            }

            return result;
        }

        pub fn dot(self: *const Self, comptime other_cols: comptime_int, other: Mat(columns, other_cols)) Mat(rows, other_cols) {
            var result = Mat(rows, other_cols){ .values = @splat(@splat(0.0)) };

            for (0..rows) |r| {
                var row_acc: @Vector(other_cols, f64) = @splat(0.0);

                inline for (0..columns) |k| {
                    const scalar = self.values[r][k];
                    row_acc += @as(@Vector(other_cols, f64), @splat(scalar)) * other.values[k];
                }

                result.values[r] = row_acc;
            }

            return result;
        }

        pub fn T(self: *const Self) Mat(columns, rows) {
            var result = Mat(columns, rows){ .values = @splat(@splat(0.0)) };

            for (0..rows) |r| {
                inline for (0..columns) |c| {
                    result.values[c][r] = self.values[r][c];
                }
            }
            return result;
        }

        pub fn clip(self: *const Self, comptime n: comptime_int) Mat(n, columns) {
            var result = Mat(n, columns){ .values = undefined };
            inline for (0..n) |i| {
                result.values[i] = self.values[i];
            }
            return result;
        }

        pub fn format(self: *const Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;
            try writer.writeAll("Mat[\n");
            for (0..rows) |r| {
                try writer.print("  {d}\n", .{self.values[r]});
            }
            try writer.writeAll("]");
        }
    };
}

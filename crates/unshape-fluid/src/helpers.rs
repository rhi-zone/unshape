use glam::Vec2;

pub(crate) fn idx2d(x: usize, y: usize, width: usize) -> usize {
    y * width + x
}

pub(crate) fn idx3d(x: usize, y: usize, z: usize, width: usize, height: usize) -> usize {
    z * width * height + y * width + x
}

pub(crate) fn bilinear_sample_2d(field: &[f32], pos: Vec2, width: usize, height: usize) -> f32 {
    let x = pos.x.clamp(0.0, (width - 1) as f32);
    let y = pos.y.clamp(0.0, (height - 1) as f32);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let sx = x - x0 as f32;
    let sy = y - y0 as f32;

    let v00 = field[idx2d(x0, y0, width)];
    let v10 = field[idx2d(x1, y0, width)];
    let v01 = field[idx2d(x0, y1, width)];
    let v11 = field[idx2d(x1, y1, width)];

    let v0 = v00 * (1.0 - sx) + v10 * sx;
    let v1 = v01 * (1.0 - sx) + v11 * sx;

    v0 * (1.0 - sy) + v1 * sy
}

pub(crate) fn diffuse_2d(
    b: i32,
    x: &mut [f32],
    x0: &[f32],
    diff: f32,
    dt: f32,
    iters: u32,
    width: usize,
    height: usize,
) {
    let a = dt * diff * (width * height) as f32;
    lin_solve_2d(b, x, x0, a, 1.0 + 4.0 * a, iters, width, height);
}

pub(crate) fn lin_solve_2d(
    b: i32,
    x: &mut [f32],
    x0: &[f32],
    a: f32,
    c: f32,
    iters: u32,
    width: usize,
    height: usize,
) {
    let c_recip = 1.0 / c;

    for _ in 0..iters {
        for j in 1..height - 1 {
            for i in 1..width - 1 {
                let idx = idx2d(i, j, width);
                x[idx] = (x0[idx]
                    + a * (x[idx2d(i + 1, j, width)]
                        + x[idx2d(i - 1, j, width)]
                        + x[idx2d(i, j + 1, width)]
                        + x[idx2d(i, j - 1, width)]))
                    * c_recip;
            }
        }
        set_bnd_2d(b, x, width, height);
    }
}

pub(crate) fn project_2d(
    vx: &mut [f32],
    vy: &mut [f32],
    p: &mut [f32],
    div: &mut [f32],
    iters: u32,
    width: usize,
    height: usize,
) {
    let h = 1.0 / width.max(height) as f32;

    for j in 1..height - 1 {
        for i in 1..width - 1 {
            let idx = idx2d(i, j, width);
            div[idx] = -0.5
                * h
                * (vx[idx2d(i + 1, j, width)] - vx[idx2d(i - 1, j, width)]
                    + vy[idx2d(i, j + 1, width)]
                    - vy[idx2d(i, j - 1, width)]);
            p[idx] = 0.0;
        }
    }

    set_bnd_2d(0, div, width, height);
    set_bnd_2d(0, p, width, height);
    lin_solve_2d(0, p, div, 1.0, 4.0, iters, width, height);

    for j in 1..height - 1 {
        for i in 1..width - 1 {
            let idx = idx2d(i, j, width);
            vx[idx] -= 0.5 * (p[idx2d(i + 1, j, width)] - p[idx2d(i - 1, j, width)]) / h;
            vy[idx] -= 0.5 * (p[idx2d(i, j + 1, width)] - p[idx2d(i, j - 1, width)]) / h;
        }
    }

    set_bnd_2d(1, vx, width, height);
    set_bnd_2d(2, vy, width, height);
}

pub(crate) fn advect_2d(
    b: i32,
    d: &mut [f32],
    d0: &[f32],
    vx: &[f32],
    vy: &[f32],
    dt: f32,
    width: usize,
    height: usize,
) {
    let dt0 = dt * width.max(height) as f32;

    for j in 1..height - 1 {
        for i in 1..width - 1 {
            let idx = idx2d(i, j, width);
            let x = (i as f32 - dt0 * vx[idx]).clamp(0.5, width as f32 - 1.5);
            let y = (j as f32 - dt0 * vy[idx]).clamp(0.5, height as f32 - 1.5);

            let i0 = x.floor() as usize;
            let i1 = i0 + 1;
            let j0 = y.floor() as usize;
            let j1 = j0 + 1;

            let s1 = x - i0 as f32;
            let s0 = 1.0 - s1;
            let t1 = y - j0 as f32;
            let t0 = 1.0 - t1;

            d[idx] = s0 * (t0 * d0[idx2d(i0, j0, width)] + t1 * d0[idx2d(i0, j1, width)])
                + s1 * (t0 * d0[idx2d(i1, j0, width)] + t1 * d0[idx2d(i1, j1, width)]);
        }
    }

    set_bnd_2d(b, d, width, height);
}

pub(crate) fn set_bnd_2d(b: i32, x: &mut [f32], width: usize, height: usize) {
    // Top and bottom boundaries
    for i in 1..width - 1 {
        x[idx2d(i, 0, width)] = if b == 2 {
            -x[idx2d(i, 1, width)]
        } else {
            x[idx2d(i, 1, width)]
        };
        x[idx2d(i, height - 1, width)] = if b == 2 {
            -x[idx2d(i, height - 2, width)]
        } else {
            x[idx2d(i, height - 2, width)]
        };
    }

    // Left and right boundaries
    for j in 1..height - 1 {
        x[idx2d(0, j, width)] = if b == 1 {
            -x[idx2d(1, j, width)]
        } else {
            x[idx2d(1, j, width)]
        };
        x[idx2d(width - 1, j, width)] = if b == 1 {
            -x[idx2d(width - 2, j, width)]
        } else {
            x[idx2d(width - 2, j, width)]
        };
    }

    // Corners
    x[idx2d(0, 0, width)] = 0.5 * (x[idx2d(1, 0, width)] + x[idx2d(0, 1, width)]);
    x[idx2d(0, height - 1, width)] =
        0.5 * (x[idx2d(1, height - 1, width)] + x[idx2d(0, height - 2, width)]);
    x[idx2d(width - 1, 0, width)] =
        0.5 * (x[idx2d(width - 2, 0, width)] + x[idx2d(width - 1, 1, width)]);
    x[idx2d(width - 1, height - 1, width)] =
        0.5 * (x[idx2d(width - 2, height - 1, width)] + x[idx2d(width - 1, height - 2, width)]);
}

pub(crate) fn diffuse_3d(
    b: i32,
    x: &mut [f32],
    x0: &[f32],
    diff: f32,
    dt: f32,
    iters: u32,
    width: usize,
    height: usize,
    depth: usize,
) {
    let a = dt * diff * (width * height * depth) as f32;
    lin_solve_3d(b, x, x0, a, 1.0 + 6.0 * a, iters, width, height, depth);
}

pub(crate) fn lin_solve_3d(
    b: i32,
    x: &mut [f32],
    x0: &[f32],
    a: f32,
    c: f32,
    iters: u32,
    width: usize,
    height: usize,
    depth: usize,
) {
    let c_recip = 1.0 / c;

    for _ in 0..iters {
        for k in 1..depth - 1 {
            for j in 1..height - 1 {
                for i in 1..width - 1 {
                    let idx = idx3d(i, j, k, width, height);
                    x[idx] = (x0[idx]
                        + a * (x[idx3d(i + 1, j, k, width, height)]
                            + x[idx3d(i - 1, j, k, width, height)]
                            + x[idx3d(i, j + 1, k, width, height)]
                            + x[idx3d(i, j - 1, k, width, height)]
                            + x[idx3d(i, j, k + 1, width, height)]
                            + x[idx3d(i, j, k - 1, width, height)]))
                        * c_recip;
                }
            }
        }
        set_bnd_3d(b, x, width, height, depth);
    }
}

pub(crate) fn project_3d(
    vx: &mut [f32],
    vy: &mut [f32],
    vz: &mut [f32],
    p: &mut [f32],
    div: &mut [f32],
    iters: u32,
    width: usize,
    height: usize,
    depth: usize,
) {
    let h = 1.0 / (width.max(height).max(depth)) as f32;

    // Calculate divergence
    for k in 1..depth - 1 {
        for j in 1..height - 1 {
            for i in 1..width - 1 {
                let idx = idx3d(i, j, k, width, height);
                div[idx] = -0.5
                    * h
                    * (vx[idx3d(i + 1, j, k, width, height)]
                        - vx[idx3d(i - 1, j, k, width, height)]
                        + vy[idx3d(i, j + 1, k, width, height)]
                        - vy[idx3d(i, j - 1, k, width, height)]
                        + vz[idx3d(i, j, k + 1, width, height)]
                        - vz[idx3d(i, j, k - 1, width, height)]);
                p[idx] = 0.0;
            }
        }
    }

    set_bnd_3d(0, div, width, height, depth);
    set_bnd_3d(0, p, width, height, depth);
    lin_solve_3d(0, p, div, 1.0, 6.0, iters, width, height, depth);

    // Subtract pressure gradient
    for k in 1..depth - 1 {
        for j in 1..height - 1 {
            for i in 1..width - 1 {
                let idx = idx3d(i, j, k, width, height);
                vx[idx] -= 0.5
                    * (p[idx3d(i + 1, j, k, width, height)] - p[idx3d(i - 1, j, k, width, height)])
                    / h;
                vy[idx] -= 0.5
                    * (p[idx3d(i, j + 1, k, width, height)] - p[idx3d(i, j - 1, k, width, height)])
                    / h;
                vz[idx] -= 0.5
                    * (p[idx3d(i, j, k + 1, width, height)] - p[idx3d(i, j, k - 1, width, height)])
                    / h;
            }
        }
    }

    set_bnd_3d(1, vx, width, height, depth);
    set_bnd_3d(2, vy, width, height, depth);
    set_bnd_3d(3, vz, width, height, depth);
}

pub(crate) fn advect_3d(
    b: i32,
    d: &mut [f32],
    d0: &[f32],
    vx: &[f32],
    vy: &[f32],
    vz: &[f32],
    dt: f32,
    width: usize,
    height: usize,
    depth: usize,
) {
    let dt0 = dt * (width.max(height).max(depth)) as f32;

    for k in 1..depth - 1 {
        for j in 1..height - 1 {
            for i in 1..width - 1 {
                let idx = idx3d(i, j, k, width, height);
                let x = (i as f32 - dt0 * vx[idx]).clamp(0.5, width as f32 - 1.5);
                let y = (j as f32 - dt0 * vy[idx]).clamp(0.5, height as f32 - 1.5);
                let z = (k as f32 - dt0 * vz[idx]).clamp(0.5, depth as f32 - 1.5);

                let i0 = x.floor() as usize;
                let i1 = i0 + 1;
                let j0 = y.floor() as usize;
                let j1 = j0 + 1;
                let k0 = z.floor() as usize;
                let k1 = k0 + 1;

                let s1 = x - i0 as f32;
                let s0 = 1.0 - s1;
                let t1 = y - j0 as f32;
                let t0 = 1.0 - t1;
                let u1 = z - k0 as f32;
                let u0 = 1.0 - u1;

                d[idx] = s0
                    * (t0
                        * (u0 * d0[idx3d(i0, j0, k0, width, height)]
                            + u1 * d0[idx3d(i0, j0, k1, width, height)])
                        + t1 * (u0 * d0[idx3d(i0, j1, k0, width, height)]
                            + u1 * d0[idx3d(i0, j1, k1, width, height)]))
                    + s1 * (t0
                        * (u0 * d0[idx3d(i1, j0, k0, width, height)]
                            + u1 * d0[idx3d(i1, j0, k1, width, height)])
                        + t1 * (u0 * d0[idx3d(i1, j1, k0, width, height)]
                            + u1 * d0[idx3d(i1, j1, k1, width, height)]));
            }
        }
    }

    set_bnd_3d(b, d, width, height, depth);
}

pub(crate) fn set_bnd_3d(b: i32, x: &mut [f32], width: usize, height: usize, depth: usize) {
    // Face boundaries
    for k in 1..depth - 1 {
        for j in 1..height - 1 {
            x[idx3d(0, j, k, width, height)] = if b == 1 {
                -x[idx3d(1, j, k, width, height)]
            } else {
                x[idx3d(1, j, k, width, height)]
            };
            x[idx3d(width - 1, j, k, width, height)] = if b == 1 {
                -x[idx3d(width - 2, j, k, width, height)]
            } else {
                x[idx3d(width - 2, j, k, width, height)]
            };
        }
    }

    for k in 1..depth - 1 {
        for i in 1..width - 1 {
            x[idx3d(i, 0, k, width, height)] = if b == 2 {
                -x[idx3d(i, 1, k, width, height)]
            } else {
                x[idx3d(i, 1, k, width, height)]
            };
            x[idx3d(i, height - 1, k, width, height)] = if b == 2 {
                -x[idx3d(i, height - 2, k, width, height)]
            } else {
                x[idx3d(i, height - 2, k, width, height)]
            };
        }
    }

    for j in 1..height - 1 {
        for i in 1..width - 1 {
            x[idx3d(i, j, 0, width, height)] = if b == 3 {
                -x[idx3d(i, j, 1, width, height)]
            } else {
                x[idx3d(i, j, 1, width, height)]
            };
            x[idx3d(i, j, depth - 1, width, height)] = if b == 3 {
                -x[idx3d(i, j, depth - 2, width, height)]
            } else {
                x[idx3d(i, j, depth - 2, width, height)]
            };
        }
    }
}

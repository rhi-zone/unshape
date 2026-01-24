# unshape-motion-fn

Motion functions for time-based animation.

## Purpose

Provides motion primitives that map time to values - the building blocks for smooth animations. Each motion function implements `Field<f32, T>` where the input is time and output is the animated value.

Also includes `MotionExpr`, a typed expression AST for motion that enables UI introspection, JSON serialization, and GPU compilation via dew.

## Related Crates

- **unshape-field** - Core `Field` trait that motion functions implement
- **unshape-easing** - Easing functions used by `Eased` motion
- **unshape-noise** - Perlin noise used by `Wiggle` motion
- **unshape-motion** - Scene graph that uses motion functions for layer animation
- **unshape-expr-field** - Similar typed AST for spatial fields (`FieldExpr`)

## Motion Functions

### Constant
Value that doesn't change over time:
```rust
let motion = Constant::new(100.0);
assert_eq!(motion.at(0.0), 100.0);
assert_eq!(motion.at(999.0), 100.0);
```

### Lerp
Linear interpolation over a duration:
```rust
let motion = Lerp::new(0.0, 100.0, 1.0); // from, to, duration
assert_eq!(motion.at(0.0), 0.0);
assert_eq!(motion.at(0.5), 50.0);
assert_eq!(motion.at(1.0), 100.0);
```

### Eased
Interpolation with easing function:
```rust
let motion = Eased::new(0.0, 100.0, 1.0, EasingType::CubicInOut);
// Smooth acceleration/deceleration
```

### Spring
Physically-based spring motion with optional overshoot:
```rust
// Critically damped (no overshoot)
let smooth = Spring::critical(0.0, 100.0, 300.0);

// Underdamped (bouncy)
let bouncy = Spring::bouncy(0.0, 100.0, 300.0);

// Custom damping
let custom = Spring::new(0.0, 100.0, 300.0, 15.0);
```

### Oscillate
Sine wave oscillation:
```rust
let motion = Oscillate::new(
    0.0,   // center
    1.0,   // amplitude
    2.0,   // frequency (Hz)
    0.0,   // phase offset
);
// Oscillates between -1 and +1 at 2Hz
```

### Wiggle
Noise-based random motion (like After Effects' wiggle):
```rust
let motion = Wiggle::new(
    0.0,   // center
    10.0,  // amplitude
    2.0,   // frequency
    42.0,  // seed
);
// Smooth random values around 0, varying ±10
```

## Combinators

Wrap any motion to modify its timing:

```rust
// Delay start by 0.5 seconds
let delayed = Delay::new(motion, 0.5);

// Speed up 2x
let fast = TimeScale::new(motion, 2.0);

// Loop every 1 second
let looped = Loop::new(motion, 1.0);

// Ping-pong (forward then backward)
let pingpong = PingPong::new(motion, 1.0);
```

## MotionExpr (Typed AST)

For UI introspection and serialization, use `MotionExpr` instead of runtime structs:

```rust
use rhi_unshape_motion_fn::MotionExpr;

// Build expression tree
let expr = MotionExpr::Add(
    Box::new(MotionExpr::Spring {
        from: Box::new(MotionExpr::Constant(0.0)),
        to: Box::new(MotionExpr::Constant(100.0)),
        stiffness: 300.0,
        damping: 15.0,
    }),
    Box::new(MotionExpr::Wiggle {
        center: 0.0,
        amplitude: 5.0,
        frequency: 2.0,
        seed: 42.0,
    }),
);

// Evaluate
let value = expr.eval(0.5, &HashMap::new());

// Inspect free variables
let vars = expr.free_vars();

// Convert to dew AST for GPU compilation
#[cfg(feature = "dew")]
let ast = expr.to_dew_ast();
```

## Dew Integration

Register motion functions for use in dew expressions:

```rust
#[cfg(feature = "dew")]
{
    let mut registry = scalar_registry();
    register_motion_functions(&mut registry);

    // Now can parse: "spring(0, 100, 300, 15, t) + wiggle(0, 5, 2, 42, t)"
}
```

## Use Cases

### UI Animation
Animate UI element positions with spring physics:
```rust
let spring = Spring::critical(old_pos, new_pos, 300.0);
// Sample at each frame time
```

### Keyframe Interpolation
Smooth transitions between keyframes:
```rust
let motion = Eased::new(key1.value, key2.value, duration, EasingType::CubicInOut);
```

### Procedural Animation
Add organic variation to motion:
```rust
let base = Spring::critical(0.0, target, 300.0);
let jitter = Wiggle::new(0.0, 2.0, 8.0, seed);
// Combine: base.at(t) + jitter.at(t)
```

### Motion Graphics
Drive After Effects-style animations:
```rust
// Position with overshoot
let x = Spring::bouncy(0.0, 500.0, 200.0);
let y = Eased::new(0.0, 300.0, 1.0, EasingType::ExpoOut);

// Rotation with continuous spin
let rotation = Lerp::new(0.0, TAU, 2.0); // Full rotation over 2s

// Scale with bounce
let scale = Spring::bouncy(0.0, 1.0, 400.0);
```

## Compositions

### With unshape-motion
Drive scene graph transforms:
```rust
let layer = scene.get_layer_mut("logo").unwrap();
layer.transform.position.x = spring_x.at(time);
layer.transform.rotation = rotation.at(time);
layer.opacity = fade.at(time);
```

### With unshape-rig
Animate skeletal properties:
```rust
// Procedural breathing
let breath = Oscillate::new(1.0, 0.05, 0.2, 0.0);
rig.set_scale("chest", breath.at(time));
```

### With unshape-scatter
Stagger instance timing:
```rust
for (i, instance) in instances.iter().enumerate() {
    let delay = i as f32 * 0.1;
    let motion = Delay::new(Spring::critical(0.0, 1.0, 300.0), delay);
    instance.scale = motion.at(time);
}
```

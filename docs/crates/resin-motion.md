# resin-motion

2D motion graphics scene graph for After Effects-style animation.

## Purpose

Provides a hierarchical scene structure for 2D motion graphics. The scene graph handles parent-child transform inheritance, opacity accumulation, and blend modes - the foundation for compositing animated vector graphics.

## Related Crates

- **resin-motion-fn** - Motion functions (Spring, Wiggle, etc.) for animating properties
- **resin-color** - `BlendMode` enum used for layer compositing
- **resin-vector** - Vector paths that can be rendered in layers
- **resin-easing** - Easing functions for animation curves

## Core Types

### Transform2D

2D transform with anchor point (pivot for rotation/scale):

```rust
let transform = Transform2D::new()
    .with_position(Vec2::new(100.0, 100.0))
    .with_anchor(Vec2::new(50.0, 50.0))  // Pivot point
    .with_rotation(45.0_f32.to_radians())
    .with_scale(2.0);

// Transform a point from local to parent space
let world_pos = transform.transform_point(Vec2::ZERO);

// Get the transformation matrix
let matrix = transform.to_matrix();
```

The anchor point defines where rotation and scaling happen. This matches After Effects behavior where the anchor is the center of transformation.

Transform order: translate(-anchor) → scale → skew → rotate → translate(+anchor) → translate(position)

### Layer

A node in the scene hierarchy:

```rust
let mut layer = Layer::new("shape")
    .with_content("path_001")  // Reference to external content
    .with_opacity(0.8)
    .with_blend_mode(BlendMode::Multiply);

layer.transform.position = Vec2::new(100.0, 50.0);
layer.transform.rotation = 0.5;

// Add children
layer.add_child(Layer::new("child1"));
layer.add_child(Layer::new("child2"));

// Find nested layers
let child = layer.get_child("child1");
```

Layers reference content by ID rather than containing it directly. This separates the scene hierarchy from the actual drawable content (paths, images, etc.).

### Scene

Root container for the layer hierarchy:

```rust
let mut scene = Scene::with_size(1920.0, 1080.0);
scene.frame_rate = 30.0;
scene.duration = 10.0;

scene.add_layer(Layer::new("background"));
scene.add_layer(Layer::new("content"));
scene.add_layer(Layer::new("foreground"));

// Find layers by path
let nested = scene.find_by_path("content/group/shape");
```

## Transform Resolution

Flatten the hierarchy into world-space transforms for rendering:

```rust
let resolved = scene.resolve_transforms();

for item in &resolved {
    // item.layer - reference to original layer
    // item.world_matrix - Mat3 from local to scene root
    // item.world_opacity - accumulated opacity
    // item.depth - nesting level

    // Transform a point to world space
    let world_pos = item.transform_point(local_pos);

    // Use for rendering
    if item.layer.content_id.is_some() {
        render_content(item.layer.content_id, item.world_matrix, item.world_opacity);
    }
}
```

Hidden layers (and their children) are excluded from resolution.

## Use Cases

### Motion Graphics Composition

Build a logo animation:

```rust
let mut scene = Scene::with_size(1920.0, 1080.0);

// Background
let mut bg = Layer::new("background");
bg.content_id = Some("gradient_001".into());
scene.add_layer(bg);

// Logo group
let mut logo = Layer::new("logo");
logo.transform.position = Vec2::new(960.0, 540.0);
logo.transform.anchor = Vec2::new(100.0, 50.0); // Center of logo

// Logo parts
let mut icon = Layer::new("icon");
icon.transform.position = Vec2::new(-60.0, 0.0);
icon.content_id = Some("icon_path".into());

let mut text = Layer::new("text");
text.transform.position = Vec2::new(60.0, 0.0);
text.content_id = Some("text_path".into());

logo.add_child(icon);
logo.add_child(text);
scene.add_layer(logo);
```

### Animating the Scene

Use motion functions to drive transforms over time:

```rust
use rhi_unshape_motion_fn::{Spring, Oscillate, Eased, Motion};

fn animate_scene(scene: &mut Scene, time: f32) {
    // Logo entrance
    let logo = scene.get_layer_mut("logo").unwrap();

    // Spring animation for position
    let spring_y = Spring::critical(800.0, 540.0, 300.0);
    logo.transform.position.y = spring_y.at(time);

    // Fade in
    let fade = Eased::new(0.0, 1.0, 0.5, EasingType::CubicOut);
    logo.opacity = fade.at(time);

    // Subtle rotation wobble
    let wobble = Oscillate::new(0.0, 0.02, 0.5, 0.0);
    logo.transform.rotation = wobble.at(time);
}
```

### Layer Iteration

Process all layers:

```rust
// Depth-first iteration
for layer in scene.iter_layers() {
    println!("{}: visible={}", layer.name, layer.visible);
}

// Count total layers
let count = scene.layer_count();
```

### Serialization

Save and load scenes:

```rust
#[cfg(feature = "serde")]
{
    // Save
    let json = serde_json::to_string_pretty(&scene)?;
    std::fs::write("scene.json", json)?;

    // Load
    let json = std::fs::read_to_string("scene.json")?;
    let scene: Scene = serde_json::from_str(&json)?;
}
```

## Transform Interpolation

Smoothly blend between transforms:

```rust
let t1 = Transform2D::from_position(Vec2::new(0.0, 0.0));
let t2 = Transform2D::from_position(Vec2::new(100.0, 100.0))
    .with_rotation(1.0)
    .with_scale(2.0);

// Interpolate at 50%
let mid = t1.lerp(&t2, 0.5);
```

## Compositions

### With resin-vector

Render vector paths in layers:

```rust
// Scene references paths by ID
layer.content_id = Some("logo_outline".into());

// Separately maintain path storage
let paths: HashMap<String, Path> = ...;

// At render time
for resolved in scene.resolve_transforms() {
    if let Some(id) = &resolved.layer.content_id {
        if let Some(path) = paths.get(id) {
            render_path(path, resolved.world_matrix, resolved.world_opacity);
        }
    }
}
```

### With resin-motion-fn

See animation example above. Motion functions drive any animatable property:
- `transform.position`, `transform.rotation`, `transform.scale`
- `opacity`
- Custom properties via content lookup

### With resin-scatter

Create instanced animations with staggered timing:

```rust
// Create instance layers
for (i, pos) in positions.iter().enumerate() {
    let mut instance = Layer::new(format!("instance_{}", i));
    instance.transform.position = *pos;
    instance.content_id = Some("shape".into());
    parent.add_child(instance);
}

// Animate with stagger
fn animate_instances(scene: &mut Scene, time: f32) {
    let parent = scene.get_layer_mut("instances").unwrap();
    for (i, child) in parent.children.iter_mut().enumerate() {
        let delay = i as f32 * 0.05;
        let spring = Delay::new(Spring::bouncy(0.0, 1.0, 300.0), delay);
        child.transform.scale = Vec2::splat(spring.at(time));
    }
}
```

## Design Notes

**Content separation**: Layers hold `content_id` strings rather than actual content. This keeps the scene graph lightweight and allows content to be managed separately (e.g., in a resource manager, with different storage strategies).

**Anchor point behavior**: Matches After Effects - the anchor defines the point around which rotation and scale occur, relative to the layer's local origin.

**Opacity accumulation**: Child opacity multiplies with parent opacity during resolution. A child at 50% opacity under a parent at 50% opacity renders at 25% effective opacity.

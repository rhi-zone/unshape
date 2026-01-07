# Winding Rules

How to determine which regions of a path are "inside" (filled) vs "outside".

## The Problem

Given a complex or self-intersecting path, which pixels should be filled?

```
   ┌─────────────────┐
   │    ┌───────┐    │
   │    │       │    │
   │    │   ?   │    │
   │    │       │    │
   │    └───────┘    │
   │                 │
   └─────────────────┘
```

Is the inner rectangle filled or a hole?

## Winding Number

For any point, draw a ray to infinity and count edge crossings:
- Edge going up: +1
- Edge going down: -1

Sum = winding number for that point.

## Even-Odd Rule

**Fill if winding number is odd.**

```rust
fn is_inside_even_odd(winding: i32) -> bool {
    winding.abs() % 2 == 1
}
```

**Behavior:**
- Ignores direction of edges
- Alternates inside/outside with each crossing
- Self-intersections create checkerboard pattern

```
Winding:  0    1    0    1    0
       ████░░░░████░░░░████
```

**Pros:**
- Simple to understand
- Direction-agnostic (path direction doesn't matter)
- Predictable for nested shapes

**Cons:**
- Can't control hole vs filled with direction
- Self-intersecting paths may not fill as expected

## Non-Zero Rule

**Fill if winding number ≠ 0.**

```rust
fn is_inside_non_zero(winding: i32) -> bool {
    winding != 0
}
```

**Behavior:**
- Clockwise paths add, counter-clockwise subtract
- Opposite directions cancel out
- More control over holes

```
Outer: CW (+1), Inner: CCW (-1)
Winding:  0    1    0    1    0
       ████████░░░░████████
              ^^^^
              hole (1 + (-1) = 0)
```

**Pros:**
- Control holes via path direction
- More intuitive for compound paths with holes
- Standard in fonts, SVG default

**Cons:**
- Direction matters (must track CW vs CCW)
- Harder to reason about for complex shapes

## Comparison

| Shape | Even-Odd | Non-Zero |
|-------|----------|----------|
| Simple closed path | Filled | Filled |
| Two nested CW paths | Inner is hole | Both filled |
| Outer CW, inner CCW | Inner is hole | Inner is hole |
| Figure-8 (self-intersecting) | Center empty | Center filled |

## Visual Examples

### Nested Squares (both CW)

```
Even-Odd:          Non-Zero:
┌───────────┐      ┌───────────┐
│███████████│      │███████████│
│███┌───┐███│      │███████████│
│███│   │███│      │███████████│
│███└───┘███│      │███████████│
│███████████│      │███████████│
└───────────┘      └───────────┘
```

### Outer CW, Inner CCW

```
Even-Odd:          Non-Zero:
┌───────────┐      ┌───────────┐
│███████████│      │███████████│
│███┌───┐███│      │███┌───┐███│
│███│   │███│      │███│   │███│
│███└───┘███│      │███└───┘███│
│███████████│      │███████████│
└───────────┘      └───────────┘
(same result - inner is hole in both)
```

### Figure-8 Self-Intersection

```
Even-Odd:          Non-Zero:
    ████             ████
  ██    ██         ████████
 █        █       ██████████
 █   OR   █       ██████████
  ██    ██         ████████
    ████             ████
    ████             ████
  ██    ██         ████████
 █        █       ██████████
 █        █       ██████████
  ██    ██         ████████
    ████             ████

Center empty      Center filled
```

## Prior Art

| System | Default | Configurable? |
|--------|---------|---------------|
| SVG | Non-Zero | Yes (`fill-rule`) |
| PostScript | Non-Zero | Yes (`eofill`) |
| HTML Canvas | Non-Zero | Yes |
| Cairo | Non-Zero | Yes |
| Skia | Non-Zero | Yes |
| Core Graphics | Non-Zero | Yes |
| OpenType fonts | Non-Zero | No |
| TrueType fonts | Non-Zero | No |

Almost everything defaults to non-zero, with even-odd as option.

## Recommendation

**Support both, default to non-zero.**

```rust
#[derive(Default)]
enum WindingRule {
    #[default]
    NonZero,
    EvenOdd,
}

struct Fill {
    color: Color,
    rule: WindingRule,
}
```

Reasoning:
1. Non-zero matches SVG, fonts, most tools
2. Even-odd is sometimes needed (legacy files, specific effects)
3. Low implementation cost to support both
4. Per-fill setting (not global) for flexibility

## Implementation Notes

Both rules use the same winding number calculation. Only the final `is_inside` test differs:

```rust
fn is_inside(winding: i32, rule: WindingRule) -> bool {
    match rule {
        WindingRule::NonZero => winding != 0,
        WindingRule::EvenOdd => winding.abs() % 2 == 1,
    }
}
```

The expensive part (ray casting, edge counting) is shared.

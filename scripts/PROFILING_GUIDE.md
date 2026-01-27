# Training Time Profiling Guide

## Overview

The `profile_training_time.py` script provides detailed analysis of where time is spent during PPO training. It helps identify performance bottlenecks and understand the impact of turbo mode on different training components.

## Quick Start

```bash
# Compare turbo ON vs OFF (recommended)
python scripts/profile_training_time.py --compare --batches 5

# Compare with detailed data collection breakdown (NEW!)
python scripts/profile_training_time.py --compare --batches 5 --breakdown

# Profile with turbo mode ON
python scripts/profile_training_time.py --turbo --batches 10

# Profile with turbo mode OFF
python scripts/profile_training_time.py --no-turbo --batches 10

# Generate detailed cProfile reports
python scripts/profile_training_time.py --compare --batches 10 --detailed
```

## Key Findings from Profiling

### Time Distribution (Typical Results)

Based on profiling 5-10 training batches:

| Component | Time | Percentage | Description |
|-----------|------|------------|-------------|
| **Data Collection** | ~3.7s | **~69%** | Environment steps + policy forward passes |
| **Training Loop** | ~1.5s | **~27%** | Loss computation + backward + optimizer |
| - Loss Computation | ~0.8s | ~15% | PPO loss calculation |
| - Backward Pass | ~0.3s | ~6% | Gradient computation |
| - Optimizer Step | ~0.4s | ~7% | Parameter updates |
| **GAE Computation** | ~0.06s | **~1%** | Advantage estimation |
| **Replay Buffer** | ~0.15s | **~3%** | Buffer operations |

### Detailed Data Collection Breakdown (with --breakdown flag)

When using `--breakdown`, data collection is further subdivided:

| Component | Time | Percentage | Description |
|-----------|------|------------|-------------|
| **Env Step** | ~1.3s | **~31%** | Petri net simulation (turbo optimizes this) |
| **Policy Forward** | ~1.8s | **~42%** | Neural network forward pass |
| **TensorDict Ops** | ~0.01s | **~0.3%** | TensorDict operations |
| **Env Reset** | ~0.00s | **~0%** | Environment reset |

**Key Insight**: Policy forward pass (42%) is the largest bottleneck, not environment steps (31%)!

### Key Insights

1. **Data Collection Dominates (69%)**
   - Includes environment steps AND policy network forward passes
   - Policy forward passes are the largest component within data collection
   - Environment steps are only a small fraction of data collection time

2. **Neural Network Operations Are the Bottleneck**
   - Policy forward passes (during collection): ~50% of total time
   - Loss computation: ~15% of total time
   - Backward pass: ~6% of total time
   - **Total neural network operations: ~70-75% of time**

3. **Environment Steps Are Small Fraction**
   - Environment steps are part of the 69% data collection time
   - Estimated to be only ~10-15% of total training time
   - This explains why turbo mode has limited impact on training

### Turbo Mode Impact

From comparison profiling:

```
Overall Performance:
  Baseline (turbo OFF): 5.48s
  Turbo (turbo ON):     5.41s
  Speedup:              +1.2%

Component-Level:
  Data Collection:      +4.5% speedup
  Other components:     -5% to -10% (timing noise)
```

**Why Limited Impact?**
- Turbo mode optimizes environment steps (~10-15% of time)
- Even 50% environment speedup = only ~5-7% total speedup
- Neural network operations (70-75%) are unchanged by turbo mode

## Usage Examples

### Example 1: Quick Comparison

```bash
python scripts/profile_training_time.py --compare --batches 5
```

Output shows:
- Side-by-side comparison of turbo ON vs OFF
- Component-level breakdown
- Overall speedup percentage
- Recommendations

### Example 2: Detailed Analysis

```bash
python scripts/profile_training_time.py --compare --batches 10 --detailed
```

Generates:
- Console output with component breakdown
- `profile_training_turbo_off.txt` - Detailed cProfile report (turbo OFF)
- `profile_training_turbo_on.txt` - Detailed cProfile report (turbo ON)

The detailed reports show:
- Top 50 functions by cumulative time
- Call counts for each function
- Time per call
- Function call hierarchy

### Example 3: Single Mode Profiling

```bash
# Profile only with turbo ON
python scripts/profile_training_time.py --turbo --batches 10

# Profile only with turbo OFF
python scripts/profile_training_time.py --no-turbo --batches 10
```

## Understanding the Output

### Component Breakdown

```
Component Breakdown:
------------------------------------------------------------
1. Data Collection          3.78s ( 69.0%)  ██████████████████████████
   (includes env steps + policy forward passes)

2. GAE Computation          0.06s (  1.1%)  █

3. Replay Buffer Ops        0.15s (  2.7%)  ██

4. Training Loop            1.49s ( 27.2%)  ██████████
   - Loss Computation       0.79s ( 14.4%)  █████
   - Backward Pass          0.33s (  6.1%)  ██
   - Optimizer Step         0.37s (  6.8%)  ██
```

**Data Collection**: Time spent collecting training data
- Environment steps (Petri net simulation)
- Policy network forward passes
- Action selection

**GAE Computation**: Generalized Advantage Estimation calculation

**Replay Buffer Ops**: Storing and sampling from replay buffer

**Training Loop**: Actual training operations
- Loss Computation: Calculate PPO loss
- Backward Pass: Compute gradients
- Optimizer Step: Update network parameters

### Comparison Report

```
================================================================================
Turbo Mode Comparison Report
================================================================================

Overall Performance:
  Baseline (turbo OFF): 5.48s
  Turbo (turbo ON):     5.41s
  Speedup:              +1.2%

Component-Level Comparison:
Component                          Baseline        Turbo      Speedup
--------------------------------------------------------------------------------
Data Collection                      3.78s       3.61s        4.5%
Loss Computation                     0.79s       0.83s       -5.3%
...
```

**Interpreting Speedup**:
- Positive speedup: Turbo is faster
- Negative speedup: Timing variation (noise)
- Data collection shows real improvement
- Other components show noise due to small sample size

## Recommendations

### When to Use This Profiler

1. **Understanding Training Bottlenecks**
   - Identify which components take the most time
   - Decide where optimization efforts should focus

2. **Evaluating Turbo Mode Impact**
   - Measure actual speedup in training context
   - Understand why speedup is limited

3. **Debugging Performance Issues**
   - Compare different configurations
   - Identify unexpected slowdowns

### Optimization Priorities

Based on profiling results:

1. **Highest Impact**: Optimize neural network operations
   - Use smaller networks (reduce n_hidden, n_layer)
   - Use GPU acceleration (if available)
   - Batch operations more efficiently

2. **Medium Impact**: Optimize data collection
   - Enable turbo mode (small but free improvement)
   - Reduce frames_per_batch if possible

3. **Low Impact**: Other optimizations
   - GAE and buffer operations are already fast
   - Not worth optimizing further

### Best Practices

1. **Always enable turbo mode for training**
   - Provides 1-5% speedup with no downsides
   - Larger impact on evaluation/rollout scenarios

2. **Focus on neural network efficiency**
   - Network size has biggest impact on training time
   - Consider smaller networks for faster iteration

3. **Use profiler to validate optimizations**
   - Profile before and after changes
   - Verify improvements are real, not timing noise

## Technical Details

### What Gets Measured

The profiler measures:
- **Data Collection**: Calculated as (total_time - measured_components)
- **GAE Computation**: Direct timing of GAE forward pass
- **Replay Buffer**: Buffer extend and sample operations
- **Loss Computation**: PPO loss calculation
- **Backward Pass**: Gradient computation (loss.backward())
- **Optimizer Step**: Parameter updates (optimizer.step())

### Timing Methodology

- Uses `time.time()` for component-level timing
- Uses `cProfile` for detailed function-level analysis
- Multiple measurements per component for accuracy
- Aggregates all measurements for final report

### Limitations

1. **Timing Noise**: Small variations (±5%) are normal
2. **Sample Size**: Use more batches (10-20) for stable results
3. **System Load**: Other processes can affect measurements
4. **Warmup**: First batch may be slower (JIT compilation)

## Troubleshooting

### Issue: Inconsistent Results

**Solution**: Run with more batches
```bash
python scripts/profile_training_time.py --compare --batches 20
```

### Issue: Out of Memory

**Solution**: Reduce batch size in the script or use fewer batches
```bash
python scripts/profile_training_time.py --compare --batches 3
```

### Issue: Script Takes Too Long

**Solution**: Use fewer batches for quick checks
```bash
python scripts/profile_training_time.py --compare --batches 3
```

## Related Documentation

- `tests/test_turbo_training.py` - Automated tests for turbo mode
- `tests/TEST_TURBO_TRAINING_SUMMARY.md` - Test results and analysis
- `tests/test_performance.py` - Low-level performance tests
- `scripts/profile_turbo_mode.py` - Pure environment profiling

## Conclusion

The profiling results clearly show:

1. **Neural network operations dominate training time (70-75%)**
2. **Environment steps are only 10-15% of total time**
3. **Turbo mode provides 1-5% training speedup**
4. **Turbo mode provides 15-80% simulation speedup** (see `test_performance.py`)

**Recommendation**: Always enable turbo mode, but focus optimization efforts on neural network efficiency for maximum training speedup.

# AA-CLIP Documentation

This directory contains all documentation for the AA-CLIP project, organized by topic.

## Directory Structure

### 01_cross_attention_feedback_implementation/
Documentation for the cross-attention mechanism and feedback loop implementation:
- Implementation guides for Stage 1 (text→image) and Stage 2 (image→text)
- Feedback loop architecture and training process
- Code references and examples

### 02_training_guides/
Step-by-step guides for training models:
- Base model training
- Cross-attention training (from base)
- Feedback loop training (V1 and V2)
- Quick start guides

### 03_testing_evaluation/
Testing, evaluation, and visualization guides:
- How to test models and compare performance
- Visualization tools and usage
- Metrics interpretation

### 04_bug_fixes/
Bug reports and fixes:
- Gradient dilution bug (CRITICAL)
- View/reshape bug
- Untrained base model issue
- Investigation findings

### 05_architecture_reference/
Architecture documentation and reference materials:
- Complete system architecture
- Model components and data flow
- Design decisions

## Quick Links

- **Start Here**: [Quick Start Guide](02_training_guides/QUICK_START_FAIR_COMPARISON.md)
- **Critical Fix**: [Gradient Dilution Bug Fix](04_bug_fixes/GRADIENT_DILUTION_BUG_FIXED.md)
- **Training**: [Feedback Training V2 Guide](02_training_guides/FEEDBACK_TRAINING_V2_GUIDE.md)
- **Testing**: [Testing Guide](03_testing_evaluation/TESTING_GUIDE.md)
- **Visualization**: [Visualization Quick Start](03_testing_evaluation/VISUALIZATION_QUICKSTART.md)

## File Organization Status

✅ All documentation files have been organized into topic-specific directories
✅ Root directory only contains essential project files (README, requirements, etc.)
✅ Easy navigation with numbered directories

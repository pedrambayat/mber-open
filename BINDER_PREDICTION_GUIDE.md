# Guide: Creating a Binder Prediction Pipeline with mber-open

This guide explains how to create a pipeline for binder prediction using the mber-open framework. There are three main approaches:

1. **Using the CLI tool** (simplest, recommended for quick runs)
2. **Using Python directly** (most flexible, recommended for notebooks/scripts)
3. **Creating a custom pipeline script** (best for programmatic use)

## Prerequisites

Make sure you have:
- Installed mber-open and mber-protocols
- Downloaded AlphaFold2 weights (run `bash download_af_weights.sh`)
- A GPU with sufficient VRAM (recommended: 32GB+, minimum: 16GB for VHH)

## Approach 1: Using the CLI Tool (Simplest)

The `mber-vhh` command-line interface is the easiest way to run binder prediction:

### Basic Usage

```bash
mber-vhh \
  --input-pdb /path/to/target.pdb \
  --output-dir ./output/my_binder_design \
  --chains A \
  --hotspots A56,A66
```

### With Settings File

Create a YAML settings file (`my_settings.yml`):

```yaml
input_pdb: /path/to/target.pdb
output_dir: ./output/my_binder_design
chains: A
hotspots: A56,A66
region: A:18-132  # Optional: specific region
masked_binder_seq: "EVQLVESGGGLVQPGGSLRLSCAASG*********WFRQAPGKEREF***********NADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYC************WGQGTLVTVSS"
```

Then run:

```bash
mber-vhh --settings my_settings.yml
```

### Interactive Mode

For guided setup:

```bash
mber-vhh --interactive
```

See the [VHH CLI documentation](./protocols/src/mber_protocols/stable/VHH_binder_design/VHH_CLI.md) for all available options.

## Approach 2: Using Python Directly (Most Flexible)

This approach gives you full control and is ideal for notebooks or custom scripts. The pipeline consists of three sequential modules:

### Step-by-Step Python Pipeline

```python
# Import required modules
from mber_protocols.stable.VHH_binder_design.config import (
    ModelConfig, LossConfig, TrajectoryConfig, 
    EnvironmentConfig, TemplateConfig, EvaluationConfig
)
from mber_protocols.stable.VHH_binder_design.template import TemplateModule
from mber_protocols.stable.VHH_binder_design.trajectory import TrajectoryModule
from mber_protocols.stable.VHH_binder_design.evaluation import EvaluationModule
from mber_protocols.stable.VHH_binder_design.state import DesignState, TemplateData

# Optional: Set up JAX compilation cache for faster model loading
import jax
import os
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.jax/jax_cache"))

# 1. Create configuration objects (with optional customizations)
template_config = TemplateConfig()
model_config = ModelConfig()
loss_config = LossConfig()
trajectory_config = TrajectoryConfig()
evaluation_config = EvaluationConfig()
environment_config = EnvironmentConfig(
    af_params_dir='~/.mber/af_params',  # Path to AlphaFold weights
    device='cuda:0',  # GPU device to use
)

# 2. Initialize the design state with your target and binder information
design_state = DesignState(
    template_data=TemplateData(
        target_id="Q9NZQ7",  # Uniprot ID or PDB ID for your target
        target_name="PDL1",  # Name of target protein
        region="A:18-132",  # Optional: specific region of target to consider
        target_hotspot_residues="A54,A56,A66,A115",  # Hotspot residues (comma-separated)
        masked_binder_seq="EVQLVESGGGLVQPGGSLRLSCAASG*********WFRQAPGKEREF***********NADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYC************WGQGTLVTVSS"
        # Masked sequence where * indicates positions to design (CDRs in this VHH example)
    )
)

# 3. Run the Template Module (prepares target structure, identifies hotspots, creates initial binder)
template_module = TemplateModule(
    template_config=template_config,
    environment_config=environment_config,
    verbose=True,
)

template_module.setup(design_state)
design_state = template_module.run(design_state)
template_module.teardown(design_state)

# 4. Run the Trajectory Module (designs binder sequences via optimization)
trajectory_module = TrajectoryModule(
    model_config=model_config,
    loss_config=loss_config,
    trajectory_config=trajectory_config,
    environment_config=environment_config,
)

trajectory_module.setup(design_state)
design_state = trajectory_module.run(design_state)
trajectory_module.teardown(design_state)

# 5. Run the Evaluation Module (evaluates designed binders)
evaluation_module = EvaluationModule(
    model_config=model_config,
    evaluation_config=evaluation_config,
    loss_config=loss_config,
    environment_config=environment_config,
)

evaluation_module.setup(design_state)
design_state = evaluation_module.run(design_state)
evaluation_module.teardown(design_state)

# 6. Save results to disk
design_state.to_dir('./output/my_binder_design')

print("Pipeline completed successfully!")
```

### Key Configuration Parameters

You can customize each module's behavior through configuration objects:

**TemplateConfig** - Controls target processing and initial binder generation:
- `folding_model`: "nbb2" or "esmfold" (default: "nbb2" for VHH)
- `sasa_threshold`: Surface accessibility threshold for hotspot selection
- `hotspot_strategy`: 'top_k', 'random', or 'none'
- `plm_model`: Protein language model (default: "esm2-650M")
- `sampling_temperature`: Temperature for sequence generation

**TrajectoryConfig** - Controls optimization process:
- `soft_iters`: Soft iteration count (default: 65)
- `temp_iters`: Temperature iteration count (default: 25)
- `hard_iters`: Hard iteration count (default: 0)
- `pssm_iters`: Position-specific scoring matrix iterations (default: 10)
- `optimizer_type`: "adam", "sgd", "schedule_free_adam", "schedule_free_sgd"
- `optimizer_learning_rate`: Learning rate (default: 0.4)
- `early_stop_iptm`: Early stopping threshold for iPTM (default: 0.7)

**LossConfig** - Controls loss function weights:
- `weights_con_inter`: Inter-chain contact weight (default: 0.5)
- `weights_pae_inter`: Inter-chain PAE weight (default: 1.0)
- `weights_hbond`: Hydrogen bond weight (default: 2.5)
- `weights_salt_bridge`: Salt bridge weight (default: 2.0)
- `weights_iptm`: iPTM weight (default: 0.1)

**EvaluationConfig** - Controls evaluation parameters:
- `monomer_folding_model`: Model for folding monomers (default: "nbb2")
- `plm_model`: Model for ESM scoring

**EnvironmentConfig** - Controls environment settings:
- `af_params_dir`: Path to AlphaFold weights directory
- `device`: CUDA device (e.g., "cuda:0")

## Approach 3: Creating a Custom Pipeline Script

You can create a reusable pipeline function based on the pattern in `protocols/src/mber_protocols/stable/VHH_binder_design/pipeline.py`:

```python
from typing import Optional
from mber_protocols.stable.VHH_binder_design.config import (
    TemplateConfig, ModelConfig, LossConfig, 
    TrajectoryConfig, EvaluationConfig, EnvironmentConfig
)
from mber_protocols.stable.VHH_binder_design.template import TemplateModule
from mber_protocols.stable.VHH_binder_design.trajectory import TrajectoryModule
from mber_protocols.stable.VHH_binder_design.evaluation import EvaluationModule
from mber_protocols.stable.VHH_binder_design.state import DesignState, TemplateData

def run_binder_prediction_pipeline(
    target_id: str,
    target_name: str,
    masked_binder_seq: str,
    output_dir: str,
    region: Optional[str] = None,
    hotspots: Optional[str] = None,
    device: str = "cuda:0",
    af_params_dir: str = "~/.mber/af_params",
    verbose: bool = True,
):
    """
    Complete binder prediction pipeline.
    
    Args:
        target_id: Uniprot ID or PDB ID for the target protein
        target_name: Name of the target protein
        masked_binder_seq: Binder sequence with * for positions to design
        output_dir: Directory to save results
        region: Optional region specification (e.g., "A:18-132")
        hotspots: Optional hotspot residues (e.g., "A54,A56,A66")
        device: CUDA device to use
        af_params_dir: Path to AlphaFold weights
        verbose: Whether to print progress messages
    
    Returns:
        DesignState: The final design state with all results
    """
    # Create configurations
    template_config = TemplateConfig()
    model_config = ModelConfig()
    loss_config = LossConfig()
    trajectory_config = TrajectoryConfig()
    evaluation_config = EvaluationConfig()
    environment_config = EnvironmentConfig(
        af_params_dir=af_params_dir,
        device=device,
    )
    
    # Initialize design state
    design_state = DesignState(
        template_data=TemplateData(
            target_id=target_id,
            target_name=target_name,
            region=region,
            target_hotspot_residues=hotspots,
            masked_binder_seq=masked_binder_seq,
        )
    )
    
    # Template module
    template_module = TemplateModule(
        template_config=template_config,
        environment_config=environment_config,
        verbose=verbose,
    )
    template_module.setup(design_state)
    design_state = template_module.run(design_state)
    template_module.teardown(design_state)
    
    # Trajectory module
    trajectory_module = TrajectoryModule(
        model_config=model_config,
        loss_config=loss_config,
        trajectory_config=trajectory_config,
        environment_config=environment_config,
    )
    trajectory_module.setup(design_state)
    design_state = trajectory_module.run(design_state)
    trajectory_module.teardown(design_state)
    
    # Evaluation module
    evaluation_module = EvaluationModule(
        model_config=model_config,
        evaluation_config=evaluation_config,
        loss_config=loss_config,
        environment_config=environment_config,
    )
    evaluation_module.setup(design_state)
    design_state = evaluation_module.run(design_state)
    evaluation_module.teardown(design_state)
    
    # Save results
    design_state.to_dir(output_dir)
    
    return design_state

# Example usage
if __name__ == "__main__":
    design_state = run_binder_prediction_pipeline(
        target_id="Q9NZQ7",
        target_name="PDL1",
        masked_binder_seq="EVQLVESGGGLVQPGGSLRLSCAASG*********WFRQAPGKEREF***********NADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYC************WGQGTLVTVSS",
        output_dir="./output/pdl1_design",
        region="A:18-132",
        hotspots="A54,A56,A66,A115",
        device="cuda:0",
    )
```

## Understanding the Pipeline Stages

### 1. Template Module
- **Purpose**: Prepares the target structure and creates initial binder template
- **Key Steps**:
  - Downloads/fetches target structure (if using Uniprot ID)
  - Identifies or uses provided hotspot residues
  - Creates truncated target structure (if needed)
  - Generates initial binder sequence from masked sequence
  - Folds initial binder structure
  - Creates combined target-binder template structure

### 2. Trajectory Module
- **Purpose**: Optimizes binder sequence to maximize binding
- **Key Steps**:
  - Sets up AlphaFold model for design
  - Runs gradient-based optimization iterations
  - Applies soft, temperature, and hard iterations
  - Generates multiple candidate sequences
  - Tracks optimization trajectory

### 3. Evaluation Module
- **Purpose**: Evaluates designed binders with comprehensive metrics
- **Key Steps**:
  - Folds each binder sequence as a complex with target
  - Folds each binder as a monomer
  - Calculates ESM scores (sequence quality metrics)
  - Computes binding metrics (iPTM, PAE, contacts, etc.)
  - Performs structure relaxation (optional)

## Output Structure

When you save the design state with `design_state.to_dir(output_dir)`, the following structure is created:

```
output_dir/
├── design_state.pickle          # Complete serialized design state
├── design_summary.yaml          # Human-readable summary
├── config_data/
│   └── config_data.json        # All configuration parameters
├── template_data/
│   ├── template_data.json      # Template configuration
│   ├── template_pdb.pdb        # Combined target-binder template
│   └── full_target_pdb.pdb     # Full target structure
├── trajectory_data/
│   ├── trajectory_data.json    # Optimization trajectory data
│   ├── best_pdb.pdb            # Best designed structure
│   ├── pssm_logits.png         # Position-specific scoring matrix
│   └── animated_trajectory.html # Interactive trajectory visualization
└── evaluation_data/
    ├── evaluation_data.json    # Evaluation results summary
    └── binders/                # Individual binder evaluation data
        ├── binder_0/           # Data for first binder
        ├── binder_1/           # Data for second binder
        └── ...
```

## Example: CD19 Binder Design

```python
from mber_protocols.stable.VHH_binder_design.config import *
from mber_protocols.stable.VHH_binder_design.template import TemplateModule
from mber_protocols.stable.VHH_binder_design.trajectory import TrajectoryModule
from mber_protocols.stable.VHH_binder_design.evaluation import EvaluationModule
from mber_protocols.stable.VHH_binder_design.state import DesignState, TemplateData

# Configuration
configs = {
    "template_config": TemplateConfig(),
    "model_config": ModelConfig(),
    "loss_config": LossConfig(),
    "trajectory_config": TrajectoryConfig(),
    "evaluation_config": EvaluationConfig(),
    "environment_config": EnvironmentConfig(
        af_params_dir='~/.mber/af_params',
        device='cuda:0',
    ),
}

# Design state for CD19
design_state = DesignState(
    template_data=TemplateData(
        target_id="P15391",  # Uniprot ID for Human CD19
        target_name="CD19",
        region="A:20-291",  # Extracellular domain
        target_hotspot_residues="A159,A163,A220,A222",  # FMC63 epitope residues
        masked_binder_seq="EVQLVESGGGLVQPGGSLRLSCAASG*********WFRQAPGKEREF***********NADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYC************WGQGTLVTVSS"
    )
)

# Run pipeline (template -> trajectory -> evaluation)
# Template
template_module = TemplateModule(configs["template_config"], configs["environment_config"])
template_module.setup(design_state)
design_state = template_module.run(design_state)
template_module.teardown(design_state)

# Trajectory
trajectory_module = TrajectoryModule(
    configs["model_config"], configs["loss_config"],
    configs["trajectory_config"], configs["environment_config"]
)
trajectory_module.setup(design_state)
design_state = trajectory_module.run(design_state)
trajectory_module.teardown(design_state)

# Evaluation
evaluation_module = EvaluationModule(
    configs["model_config"], configs["evaluation_config"],
    configs["loss_config"], configs["environment_config"]
)
evaluation_module.setup(design_state)
design_state = evaluation_module.run(design_state)
evaluation_module.teardown(design_state)

# Save
design_state.to_dir('./output/cd19_design')
```

## Tips and Best Practices

1. **Masked Sequence Format**: Use `*` to indicate positions that should be designed. Fixed positions should contain standard amino acid codes.

2. **Hotspot Selection**: 
   - Manually specify hotspots if you know the binding site
   - Use `sasa_threshold` to automatically select surface residues
   - Format: comma-separated chain:residue pairs (e.g., "A54,A56,A66")

3. **GPU Memory**: 
   - VHH designs work well on 16GB+ GPUs
   - Larger binders may require 32GB+ GPUs
   - Adjust batch sizes in configs if memory is limited

4. **Iteration Counts**: 
   - More iterations generally improve quality but take longer
   - Start with defaults and adjust based on results
   - `early_stop_iptm` can save time for promising designs

5. **Model Selection**:
   - `nbb2` (NanoBodyBuilder2) is optimized for VHH/nanobodies
   - `esmfold` is a general-purpose folding model
   - Use `nbb2` for VHH designs (default in VHH protocol)

6. **JAX Compilation Cache**: Setting up JAX cache significantly speeds up model loading on subsequent runs.

## Troubleshooting

- **Out of Memory**: Reduce batch sizes or use a smaller model variant
- **Slow Performance**: Enable JAX compilation cache, use GPU, check device configuration
- **Poor Results**: Adjust loss weights, increase iterations, check hotspot selection
- **Import Errors**: Ensure `pip install -e protocols` was run successfully

## Additional Resources

- [Main README](./README.md) - Framework overview
- [Protocols Guide](./protocols/README.md) - Creating custom protocols
- [Example Notebooks](./notebooks/) - Working examples
- [VHH CLI Documentation](./protocols/src/mber_protocols/stable/VHH_binder_design/VHH_CLI.md) - CLI reference


from mber_protocols.stable.VHH_binder_design.config import (
    ModelConfig, LossConfig, TrajectoryConfig, 
    EnvironmentConfig, TemplateConfig, EvaluationConfig
)
from mber_protocols.stable.VHH_binder_design.template import TemplateModule
from mber_protocols.stable.VHH_binder_design.trajectory import TrajectoryModule
from mber_protocols.stable.VHH_binder_design.evaluation import EvaluationModule
from mber_protocols.stable.VHH_binder_design.state import DesignState, TemplateData

import jax
import os
import time
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.jax/jax_cache"))

# configuration objects - customize by passing parameters
# TemplateConfig: Controls target processing and initial binder generation
template_config = TemplateConfig(
    folding_model="nbb2",  # "nbb2" (nanobody) or "esmfold" (general)
    sasa_threshold=50.0,  # Surface accessibility threshold for hotspot selection
    hotspot_strategy='random',  # 'top_k', 'random', or 'none'
    plm_model="esm2-650M",  # Protein language model
    sampling_temperature=0.1,  # Temperature for sequence generation (lower = more conservative)
    bias_temperature=1.0,  # Temperature for position-specific bias
    omit_amino_acids="C",  # Amino acids to avoid (e.g., "C" to avoid cysteines)
    target_chain="A",  # Target protein chain ID
    binder_chain="H",  # Binder chain ID
)

# ModelConfig: Controls which AlphaFold models to use
model_config = ModelConfig(
    design_models=[1, 2, 3, 4],  # Which AF models to use during design
    use_multimer_design=True,  # Use multimer model for design
    num_recycles_design=3,  # Number of recycling iterations during design
    eval_models=[0],  # Which AF models to use during evaluation
    use_multimer_eval=True,  # Use multimer model for evaluation
    num_recycles_eval=3,  # Number of recycling iterations during evaluation
)

# LossConfig: Controls loss function weights (higher = more important)
loss_config = LossConfig(
    weights_con_inter=0.5,  # Inter-chain contact weight
    weights_pae_inter=1.0,  # Inter-chain PAE (predicted aligned error) weight
    weights_hbond=2.5,  # Hydrogen bond weight
    weights_salt_bridge=2.0,  # Salt bridge weight
    weights_iptm=0.1,  # Interface predicted TM-score weight
    weights_plddt=0.1,  # Predicted LDDT weight
    weights_rg=0.3,  # Radius of gyration weight
    inter_contact_distance=20.0,  # Distance threshold for inter-chain contacts
    inter_contact_number=2,  # Number of inter-chain contacts to consider
)

# TrajectoryConfig: Controls optimization process
trajectory_config = TrajectoryConfig(
    soft_iters=65,  # Soft iteration count (more = better but slower)
    temp_iters=25,  # Temperature iteration count
    hard_iters=0,  # Hard iteration count (usually 0 for VHH)
    pssm_iters=10,  # Position-specific scoring matrix iterations
    greedy_tries=10,  # Number of greedy sequence attempts
    early_stop_iptm=0.7,  # Early stopping threshold for iPTM (0.0-1.0)
    early_stopping=True,  # Enable early stopping
    optimizer_type="schedule_free_sgd",  # "adam", "sgd", "schedule_free_adam", "schedule_free_sgd"
    optimizer_learning_rate=0.4,  # Learning rate (higher = faster but less stable)
    optimizer_b1=0.9,  # Beta1 for Adam/schedule_free optimizers
    optimizer_b2=0.999,  # Beta2 for Adam optimizer
    rm_aa="C",  # Amino acids to remove/avoid
)

# EvaluationConfig: Controls evaluation parameters
evaluation_config = EvaluationConfig(
    monomer_folding_model="nbb2",  # "nbb2" or "esmfold" for folding monomers
    plm_model="esm2-650M",  # Protein language model for ESM scoring
    use_gpu=False,  # Use GPU for AMBER relaxation (if available)
    max_iterations=0,  # Max iterations for AMBER relaxation (0 = disabled)
    tolerance=2.39,  # Tolerance for AMBER relaxation
    stiffness=10.0,  # Stiffness for AMBER relaxation
)

# EnvironmentConfig: Controls environment settings
environment_config = EnvironmentConfig(
    af_params_dir='~/.mber/af_params',  # Path to AlphaFold weights
    device='cuda:0',  # GPU device to use (e.g., 'cuda:0', 'cuda:1', or 'cpu')
)

# design state
design_state = DesignState(
    template_data=TemplateData(
        target_id="P15391", # Uniprot ID for Human CD19
        target_name="CD19",
        region="A:20-291", # Extracellular domain of CD19
        target_hotspot_residues="A54,A56,A66,A115",
        masked_binder_seq="EVQLVESGGGLVQPGGSLRLSCAASG*********WFRQAPGKEREF***********NADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYC************WGQGTLVTVSS"
    )
)

# Start timing
start_time = time.time()
print(f"Pipeline started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

# template module
template_module = TemplateModule(
    template_config=template_config,
    environment_config=environment_config,
    verbose=True
)

template_module.setup(design_state)
design_state = template_module.run(design_state)
template_module.teardown(design_state)

# trajectory module
trajectory_module = TrajectoryModule(
    model_config=model_config,
    loss_config=loss_config,
    trajectory_config=trajectory_config,
    environment_config=environment_config,
)

trajectory_module.setup(design_state)
design_state = trajectory_module.run(design_state)
trajectory_module.teardown(design_state)

# evaluation module
evaluation_module = EvaluationModule(
    model_config=model_config,
    evaluation_config=evaluation_config,
    loss_config=loss_config,
    environment_config=environment_config,
)

evaluation_module.setup(design_state)
design_state = evaluation_module.run(design_state)
evaluation_module.teardown(design_state)

# save design state
design_state.to_dir("cd19-mber")

# End timing and print results
end_time = time.time()
elapsed_time = end_time - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

print("\n" + "="*60)
print("Design completed successfully!")
print(f"Total execution time: {hours:02d}:{minutes:02d}:{seconds:02d} ({elapsed_time:.2f} seconds)")
print(f"Pipeline finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)
import os
import yaml
import subprocess
import itertools

def generate_ablation_configs():
    combinations = set()
    
    # 1. Just changing p initially (n_layers=3, n_heads=2)
    # We already have results for p=2, so we try 3, 5, 7
    for p in [3, 5, 7]:
        combinations.add((p, 3, 2))
        
    # 2. For each p value (3,5,7), change n_layers to 6, 9
    for p in [3, 5, 7]:
        for l in [6, 9]:
            combinations.add((p, l, 2))
            
    # 3. For p in 3,5 and n_layers in 3,6, try different n_heads (4,6)
    for p in [3, 5]:
        for l in [3, 6]:
            for h in [4, 6]:
                combinations.add((p, l, h))
                
    # Sort for deterministic execution
    return sorted(list(combinations))

def main():
    base_config_path = "configs/burgers1d.yaml"
    ablations_dir = "configs/ablations"
    os.makedirs(ablations_dir, exist_ok=True)
    
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
        
    # Ensure baseline targets 500 epochs
    base_config['training']['epochs'] = 500
    
    unique_combinations = generate_ablation_configs()
    print(f"Total unique ablation configurations to run: {len(unique_combinations)}\n")
    
    for idx, (p, n_layers, n_heads) in enumerate(unique_combinations, 1):
        # 1. Prepare tailored config
        exp_name = f"Ablation_Burgers1D_p{p}_L{n_layers}_H{n_heads}"
        print(f"--- Starting Ablation {idx}/{len(unique_combinations)}: {exp_name} ---")
        
        cfg = yaml.safe_load(yaml.dump(base_config))  # Deep copy
        cfg['experiment_name'] = exp_name
        cfg['model']['p'] = p
        cfg['model']['n_layers'] = n_layers
        cfg['model']['n_heads'] = n_heads
        
        # A mathematically coherent d_model for n_heads in (2, 4, 6) 
        # is the Least Common Multiple (LCM). LCM(2,4,6) = 12.
        # We set d_model = 48 so it's safely divisible by any head count.
        cfg['model']['d_model'] = 48 
        
        config_out_path = os.path.join(ablations_dir, f"ablation_p{p}_L{n_layers}_H{n_heads}.yaml")
        
        with open(config_out_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
            
        # 2. Spawn fully isolated subprocess for robust memory management per run
        cmd = [
            "python", "-u", "examples/train_burgers.py", 
            "--config", config_out_path
        ]
        
        # Subprocess ensures CUDA memory clears between entirely different architectures
        try:
            subprocess.run(cmd, check=True)
            print(f"--- Finished Ablation {idx}/{len(unique_combinations)} ---\n")
        except subprocess.CalledProcessError as e:
            print(f"--- FAILED Ablation {exp_name} with error code {e.returncode}. Skipping to next. ---\n")
        except KeyboardInterrupt:
            print("\nAborting entire ablation cascade at user request...")
            break

if __name__ == "__main__":
    main()

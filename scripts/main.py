if __name__ == "__main__":
        import sys
        import os
        import yaml
        import subprocess
  
        # Add the parent folder to the PYTHONPATH
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

        from src.training.train_ import train_model

        # === Load global config ===
        with open("config/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        # === Step 1 : Preprocessing ===
        print("Step 1: Preprocessing data")
        try:
            subprocess.run(["python", "scripts/preprocess.py"], check=True)
        except subprocess.CalledProcessError:
            print("Error during preprocessing.")
            sys.exit(1)
        print(" Step 2: Splitting train/val data...")
        try:
            subprocess.run(["python", "scripts/split.py"], check=True)
        except subprocess.CalledProcessError:
            print(" Error while splitting the data.")
            sys.exit(1)

        # === Étape 3 : Entraînement ===
        print("Step 3: Training the model..")
        try:
            
                train_model(cfg["dataset"])
        except Exception as e:
            print(f" Error during training: {e}")
            sys.exit(1)

      

        print("Full pipeline executed successfully")

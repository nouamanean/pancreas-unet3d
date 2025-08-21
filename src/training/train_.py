# File: src/training/train_.py
def train_model(cfg):
    import os, sys, time
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader

    # Ensure project imports
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.data.pancreas_dataset import PancreasPatchDataset
    from src.models.unet3D import UNet3D
    # from src.training.eveluate import evaluate_model  # activate if you use it

    print("Starting UNet3D model training...")

    # Device and performance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    torch.backends.cudnn.benchmark = True

    # Check required data folders
    required_dirs = [
        "data/processed/patches/irm_t2",
        "data/processed/patches/labels",
        "data/processed/patches",
    ]
    for d in required_dirs:
        if not os.path.exists(d):
            print(f"Missing folder: {d}")
            print("Please run preprocessing and splitting first.")
            sys.exit(1)

    # Paths from cfg (dataset section)
    train_csv = cfg["train_metadata"]
    val_csv   = cfg["val_metadata"]

    if not os.path.exists(train_csv):
        print(f"Missing file: {train_csv}")
        sys.exit(1)
    if not os.path.exists(val_csv):
        print(f"Missing file: {val_csv}")
        sys.exit(1)

    # Datasets / Loaders
    try:
        print("Loading datasets...")
        train_dataset = PancreasPatchDataset(
            patch_dir_img=cfg["patch_dir_img"],
            patch_dir_label=cfg["patch_dir_label"],
            metadata_path=cfg["train_metadata"],
        )
        val_dataset = PancreasPatchDataset(
            patch_dir_img=cfg["patch_dir_img"],
            patch_dir_label=cfg["patch_dir_label"],
            metadata_path=cfg["val_metadata"],
        )
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Validation dataset: {len(val_dataset)} samples")

        # DataLoader options (CUDA friendly)
        dl_opts = dict(batch_size=cfg["batch_size"], pin_memory=(device.type == "cuda"))
        num_workers = int(cfg.get("num_workers", 0))
        if num_workers > 0:
            dl_opts.update(num_workers=num_workers, persistent_workers=True)

        train_loader = DataLoader(train_dataset, shuffle=True,  **dl_opts)
        val_loader   = DataLoader(val_dataset,   shuffle=False, **dl_opts)
    except Exception as e:
        print(f"Error while loading datasets: {e}")
        sys.exit(1)

    # Model / Optimizer / Loss
    try:
        print("Initializing UNet3D model...")
        model = UNet3D(n_channels=1, n_classes=1).to(device)
        print("Model created successfully")

        criterion = nn.BCELoss()  # model already outputs sigmoid
        optimizer = optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
        num_epochs = int(cfg["num_epochs"])
    except Exception as e:
        print(f"Error initializing model/optimizer: {e}")
        sys.exit(1)

    # Save directories
    ckpt_dir  = os.path.join("results", "checkpoints")
    best_path = os.path.join("results", "best_model.pth")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val = float("inf")

    # Training
    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0

            for batch_idx, (img, mask) in enumerate(train_loader):
                try:
                    t0 = time.time()
                    img, mask = img.to(device), mask.to(device)

                    optimizer.zero_grad()
                    out = model(img)
                    loss = criterion(out, mask)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                    if batch_idx % 10 == 0:
                        print(f"[Epoch {epoch+1}/{num_epochs}] "
                              f"Batch {batch_idx}/{len(train_loader)} - "
                              f"Loss: {loss.item():.4f} Time: {time.time() - t0:.2f}s")
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue

            avg_loss = total_loss / max(1, len(train_loader))
            print(f"[Epoch {epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

            # Validation at each epoch
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for img, mask in val_loader:
                    img, mask = img.to(device), mask.to(device)
                    out = model(img)
                    val_loss += criterion(out, mask).item()
            val_loss /= max(1, len(val_loader))
            print(f"Validation - Loss: {val_loss:.4f}")

            # Saving
            # 1) checkpoint of the epoch
            epoch_ckpt = os.path.join(ckpt_dir, f"unet3d_epoch{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "cfg": cfg,
            }, epoch_ckpt)
            print(f"Checkpoint saved: {epoch_ckpt}")

            # 2) best model
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_loss,
                    "val_loss": val_loss,
                    "cfg": cfg,
                }, best_path)
                print(f"New best model saved: {best_path} (val_loss={best_val:.4f})")

        print("Training completed.")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Critical error during training: {e}")
        sys.exit(1)

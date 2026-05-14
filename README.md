# cellpose-sam-automation

Automated multichannel cell segmentation using Cellpose-SAM,
structured after the MONAI spleen segmentation tutorial.

## Install

[200~cat > cellpose_sam_automation.py << 'EOF'
#!/usr/bin/env python3
"""Cellpose-SAM automation — placeholder. Real script to follow."""
import argparse, sys
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--nuc-channel", type=int, default=0)
    args = p.parse_args()
    print(f"Would segment {args.input} -> {args.output}, ch={args.nuc_channel}")
if __name__ == "__main__":
    sys.exit(main())
EOF~
cat > cellpose_sam_automation.py << 'EOF'
#!/usr/bin/env python3
"""Cellpose-SAM automation — placeholder. Real script to follow."""
import argparse, sys
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--nuc-channel", type=int, default=0)
    args = p.parse_args()
    print(f"Would segment {args.input} -> {args.output}, ch={args.nuc_channel}")
if __name__ == "__main__":
    sys.exit(main())

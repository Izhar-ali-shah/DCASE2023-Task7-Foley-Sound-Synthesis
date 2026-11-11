# python run_generate.py --all --quantity 100

import argparse
from generate import label_list, samplying_sound


def main():
    parser = argparse.ArgumentParser(
        description="Convenience wrapper around generate.samplying_sound()."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available label IDs and exit.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate for all labels (overrides --id).",
    )
    parser.add_argument(
        "--id",
        type=int,
        default=0,
        help="Event ID to generate (0..6). Ignored if --all or --list is given.",
    )
    parser.add_argument(
        "--quantity",
        type=int,
        default=5,
        help="How many samples per label to generate.",
    )
    args = parser.parse_args()

    # Just list labels
    if args.list:
        print("Available labels:")
        for idx, name in enumerate(label_list):
            print(f"  {idx}: {name}")
        return

    # Generate for all labels
    if args.all:
        print("Generating for ALL labels:")
        for idx, name in enumerate(label_list):
            print(f"\n=== [{idx}] {name} ===")
            samplying_sound(idx, quantity=args.quantity)
            print(f"Done {args.quantity} samples for '{name}'.")
        print("\nAll labels done.")
        return

    # Single-label mode
    if args.id < 0 or args.id >= len(label_list):
        raise SystemExit(f"--id must be between 0 and {len(label_list)-1}, got {args.id}")

    label = label_list[args.id]
    print("\nUsing mapping:")
    for idx, name in enumerate(label_list):
        print(f"  {idx}: {name}")
    print()

    print(
        f"Generating {args.quantity} sample(s) for id={args.id} ({label}) ...",
        flush=True,
    )
    results = samplying_sound(args.id, quantity=args.quantity)

    print(f"\nDone. Generated {len(results)} sample(s) for '{label}'.")
    print("Audio files are saved under:")
    print(f"  results/{label}/")
    print("Mel-spectrograms are saved under:")
    print(f"  results_mel/{label}/\n")
    print("For example, you can run:")
    print(f"  find results/{label} -maxdepth 2 -type f -name '*.wav' | sort")
    print(f"  find results_mel/{label} -maxdepth 2 -type f -name '*.npy' | sort")


if __name__ == "__main__":
    main()

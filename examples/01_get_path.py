from synriard import get_model_path
import argparse


def main(args):
    print(f"Robot name: {args.name}")
    print(f"Robot version: {args.version}")
    print(f"Variant: {args.variant}")
    urdf_path = get_model_path(args.name, version=args.version, variant=args.variant)
    print(f"URDF path: {urdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="Alicia_D")
    parser.add_argument("--version", type=str, default="v5_6")
    parser.add_argument("--variant", type=str, default="gripper_100mm")
    args = parser.parse_args()

    main(args)

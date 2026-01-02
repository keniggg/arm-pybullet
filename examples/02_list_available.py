from synriard import list_available_models
import argparse


def main(args):
    """List all available robot models in the specified format."""
    print(f"Listing available {args.format.upper()} models:\n")
    table = list_available_models(model_format=args.format, show_path=args.show_path)
    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List all available robot models in table format"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="urdf",
        choices=["urdf", "mjcf"],
        help="Model format to list: 'urdf' or 'mjcf' (default: 'urdf')"
    )
    parser.add_argument(
        "--show_path",
        action="store_true",
        help="Show file path column in the table"
    )
    args = parser.parse_args()

    main(args)


"""Synria Robotics Robot Descriptions Package

Copyright (c) 2025 Synria Robotics Co., Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""



__version__ = "1.0.0"
__author__ = "Synria Robotics Team"
__copyright__ = "Copyright (c) 2025 Synria Robotics Co., Ltd."
__license__ = "GPL-3.0"


# Export subpackages, allowing access via `synriard.urdf` / `synriard.mjcf` / `synriard.meshes`
from . import urdf  # noqa: F401
from . import mjcf  # noqa: F401
from . import meshes  # noqa: F401


def get_model_path(name, version=None, variant=None, model_format="urdf"):
    """Get robot model file path.

    :param name: Robot name, e.g., 'Alicia_D', 'Alicia_M', 'Bessica_D'
    :param version: Robot version, e.g., 'v5_5', 'v5_6', 'v1_0'. Can be omitted for some robots
    :param variant: Variant name, e.g., '100mm', '50mm' (for Alicia_D gripper), or 'Covered', 'Skeleton' (for Bessica_D)
    :param model_format: Model format, 'urdf' or 'mjcf', default is 'urdf'
    :return: Absolute path to the model file
    """
    if model_format == "urdf":
        model_module = urdf
    elif model_format == "mjcf":
        model_module = mjcf
    else:
        raise ValueError(f"Unsupported model format: {model_format}. Use 'urdf' or 'mjcf'.")

    # Build version module name
    if version:
        version_module_name = f"{name}_{version}"
    else:
        version_module_name = name

    # Get version module
    try:
        version_module = getattr(model_module, version_module_name)
    except AttributeError:
        # Get all available robots and versions
        available_info = list_available_models(model_format=model_format, show_path=True)
        raise ValueError(
            f"Robot not found: {version_module_name}.\n\n"
            f"Available models:\n{available_info}"
        )

    # Build variant object name
    # All models follow {name}_{version}_{variant} naming convention
    if name in {"Alicia_D", "Alicia_M", "Bessica_D"} and version and variant:
        variant_obj_name = f"{name}_{version}_{variant}"
    else:
        variant_obj_name = None

    # Get variant object
    if variant_obj_name:
        try:
            variant_obj = getattr(version_module, variant_obj_name)
        except AttributeError:
            # Filter available variants from version module
            excluded_attrs = {'os', 'SimpleNamespace', 'types', 'abspath', 'dirname', 'join', '__builtins__',
                              '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_MODULE_PATH'}
            available_variants = [attr for attr in dir(version_module)
                                  if not attr.startswith('_') and attr not in excluded_attrs]
            # Get all available models for better error message
            available_info = list_available_models(model_format=model_format, show_path=True)
            raise ValueError(
                f"Variant not found: {variant_obj_name} for {name} {version}.\n\n"
                f"Available variants for {version_module_name}: {available_variants}\n\n"
                f"All available models:\n{available_info}"
            )
    else:
        # For cases without explicit variant, try to use version module directly
        variant_obj = version_module

    # Get model path
    # Map model format to actual attribute name (mjcf uses 'xml')
    format_attr = 'xml' if model_format == 'mjcf' else model_format
    try:
        model_path = getattr(variant_obj, format_attr)
    except AttributeError:
        # Get all available models for better error message
        available_info = list_available_models(model_format=model_format, show_path=True)
        raise ValueError(
            f"Model format '{model_format}' not found for {name} "
            f"{f'version {version}' if version else ''} "
            f"{f'variant {variant}' if variant else ''}.\n\n"
            f"All available models:\n{available_info}"
        )

    return model_path


def list_available_models(model_format="urdf", show_path=False):
    """List all available robot models in a table format.

    :param model_format: Model format to list, 'urdf' or 'mjcf', default is 'urdf'
    :param show_path: If True, include file path column in the table
    :return: Formatted string table showing name, version, variant, and optionally path
    """
    if model_format == "urdf":
        model_module = urdf
    elif model_format == "mjcf":
        model_module = mjcf
    else:
        raise ValueError(f"Unsupported model format: {model_format}. Use 'urdf' or 'mjcf'.")

    # Collect all models
    models = []

    # Get all version modules
    version_modules = [attr for attr in dir(model_module) if not attr.startswith('_')]

    for version_module_name in version_modules:
        try:
            version_module = getattr(model_module, version_module_name)

            # Parse robot name and version from module name
            # Format: {RobotName}_{version}, e.g., "Alicia_D_v5_5"
            parts = version_module_name.split('_')
            if len(parts) >= 3:
                # Try to find version pattern (v followed by numbers)
                version_idx = None
                for i in range(1, len(parts)):
                    if parts[i].startswith('v') and parts[i][1:].replace('_', '').isdigit():
                        version_idx = i
                        break

                if version_idx:
                    name = '_'.join(parts[:version_idx])
                    version = '_'.join(parts[version_idx:])
                else:
                    name = parts[0]
                    version = '_'.join(parts[1:])
            else:
                name = parts[0]
                version = '_'.join(parts[1:]) if len(parts) > 1 else None

            # Get all variant objects from version module
            # Filter out standard library/module attributes
            excluded_attrs = {'os', 'SimpleNamespace', 'types', 'abspath', 'dirname', 'join', '__builtins__',
                              '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_MODULE_PATH'}
            variant_attrs = [attr for attr in dir(version_module)
                             if not attr.startswith('_') and attr not in excluded_attrs]

            for variant_attr in variant_attrs:
                try:
                    variant_obj = getattr(version_module, variant_attr)
                    # Skip if it's a module or standard library object
                    if isinstance(variant_obj, type) or hasattr(variant_obj, '__module__') and variant_obj.__module__ in ('types', 'os', 'builtins'):
                        continue
                    # Map model format to actual attribute name (mjcf uses 'xml')
                    format_attr = 'xml' if model_format == 'mjcf' else model_format
                    # Check if this variant object has the requested format
                    if hasattr(variant_obj, format_attr):
                        # Get path if requested
                        path = getattr(variant_obj, format_attr) if show_path else None

                        # Extract variant name
                        # All models follow {name}_{version}_{variant} naming convention
                        if name == "Alicia_D":
                            # Format: {name}_{version}_gripper_{variant}
                            pattern = f"{name}_{version}_gripper_"
                            if variant_attr.startswith(pattern):
                                variant = variant_attr.replace(pattern, "")
                                variant = f"gripper_{variant}"  # Keep gripper_ prefix for variant
                            else:
                                variant = None
                        elif name == "Alicia_M":
                            # Format: {name}_{version}_gripper_{variant}
                            pattern = f"{name}_{version}_gripper_"
                            if variant_attr.startswith(pattern):
                                variant = variant_attr.replace(pattern, "")
                                variant = f"gripper_{variant}"  # Keep gripper_ prefix for variant
                            else:
                                variant = None
                        elif name == "Bessica_D":
                            # Format: {name}_{version}_{variant}
                            pattern = f"{name}_{version}_"
                            if variant_attr.startswith(pattern):
                                variant = variant_attr.replace(pattern, "")
                            else:
                                variant = None
                        else:
                            variant = None

                        model_entry = {
                            'name': name,
                            'version': version,
                            'variant': variant or '-',
                            'format': model_format
                        }
                        if show_path:
                            model_entry['path'] = path
                        models.append(model_entry)
                except (AttributeError, TypeError):
                    continue

        except (AttributeError, TypeError):
            continue

    # Sort models by name, version, variant
    models.sort(key=lambda x: (x['name'], x['version'] or '', x['variant']))

    # Format as table
    if not models:
        return f"No {model_format.upper()} models found."

    # Calculate column widths
    col_widths = {
        'name': max(len('Robot Name'), max(len(m['name']) for m in models)),
        'version': max(len('Version'), max(len(m['version'] or '-') for m in models)),
        'variant': max(len('Variant'), max(len(m['variant']) for m in models)),
    }

    if show_path:
        col_widths['path'] = max(len('Path'), max(len(m.get('path', '')) for m in models))

    # Build table header
    if show_path:
        header = f"{'Robot Name':<{col_widths['name']}} | {'Version':<{col_widths['version']}} | {'Variant':<{col_widths['variant']}} | {'Path':<{col_widths['path']}}"
        separator = f"{'-' * col_widths['name']}-+-{'-' * col_widths['version']}-+-{'-' * col_widths['variant']}-+-{'-' * col_widths['path']}"
    else:
        header = f"{'Robot Name':<{col_widths['name']}} | {'Version':<{col_widths['version']}} | {'Variant':<{col_widths['variant']}}"
        separator = f"{'-' * col_widths['name']}-+-{'-' * col_widths['version']}-+-{'-' * col_widths['variant']}"

    lines = [header, separator]
    for model in models:
        if show_path:
            line = f"{model['name']:<{col_widths['name']}} | {model['version'] or '-':<{col_widths['version']}} | {model['variant']:<{col_widths['variant']}} | {model.get('path', ''):<{col_widths['path']}}"
        else:
            line = f"{model['name']:<{col_widths['name']}} | {model['version'] or '-':<{col_widths['version']}} | {model['variant']:<{col_widths['variant']}}"
        lines.append(line)

    return '\n'.join(lines)




__all__ = [
    "urdf",
    "mjcf",
    "meshes",
    "get_model_path",
    "list_available_models",
    "__version__",
    "__author__",
    "__license__",
]

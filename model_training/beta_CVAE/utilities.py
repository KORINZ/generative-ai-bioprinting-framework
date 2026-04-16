import os


def create_output_directory(output_dir: str) -> str:
    """Create the output directory if it does not exist.
    Args:
        output_dir (str): The path to the output directory.
    Returns:
        str: The path to the created output directory.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

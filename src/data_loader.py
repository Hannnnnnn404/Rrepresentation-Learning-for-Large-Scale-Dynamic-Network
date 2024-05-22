
def parse_as_path(line):
    """
    Parse the AS path from a BGP message.

    Parameters:
    line (str): A line from the BGP message file.

    Returns:
    list: A list of AS numbers representing the AS path, or an empty list if the AS path is not present.
    """
    parts = line.strip().split('|')
    if len(parts) > 6:
        return parts[6].split()  # AS PATH是空格分隔
    return []

"""Lean 4 verification utilities."""

import re
from lean_interact import LeanREPLConfig, AutoLeanServer, Command


def extract_lean_code(llm_response):
    """Extract Lean code from LLM response.

    Looks for code wrapped in <lean></lean> tags or ```lean``` markdown blocks.

    Args:
        llm_response: The LLM's text response

    Returns:
        str or None: Extracted Lean code, or None if not found
    """
    # Try XML-style tags first
    match = re.search(r'<lean>(.*?)</lean>', llm_response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: try markdown code blocks
    code_blocks = re.findall(r'```lean\s*(.*?)```', llm_response, re.DOTALL | re.IGNORECASE)
    if code_blocks:
        return '\n\n'.join(block.strip() for block in code_blocks)

    return None


def verify_with_lean(lean_code, lean_server, verbose=False):
    """Verify Lean code and return verification results.

    Args:
        lean_code: The Lean code to verify
        lean_server: LeanServer instance
        verbose: Whether to print verbose output

    Returns:
        dict: Verification results with keys:
            - success (bool): Whether verification succeeded
            - env: Lean environment (if successful)
            - errors (list): List of error messages
            - warnings (list): List of warning messages
            - all_messages (list): All messages from Lean
    """
    try:
        if verbose:
            print(f"\nVerifying Lean code:\n{lean_code}\n")

        response = lean_server.run(Command(cmd=lean_code))

        messages = response.messages if hasattr(response, 'messages') else []
        errors = [msg for msg in messages if msg.severity == 'error']
        warnings = [msg for msg in messages if msg.severity == 'warning']

        success = len(errors) == 0

        result = {
            'success': success,
            'env': response.env if hasattr(response, 'env') else None,
            'errors': [msg.data for msg in errors],
            'warnings': [msg.data for msg in warnings],
            'all_messages': [{'severity': msg.severity, 'data': msg.data} for msg in messages]
        }

        if verbose:
            print(f"Verification {'succeeded' if success else 'failed'}")
            if errors:
                print(f"Errors: {errors}")

        return result

    except Exception as e:
        return {
            'success': False,
            'env': None,
            'errors': [str(e)],
            'warnings': [],
            'all_messages': []
        }


async def verify_with_lean_async(lean_code, lean_server, verbose=False, timeout=60.0):
    """Verify Lean code asynchronously using AutoLeanServer.async_run().

    This is more efficient in async contexts as it doesn't block the event loop.

    Args:
        lean_code: The Lean code to verify
        lean_server: AutoLeanServer instance
        verbose: Whether to print verbose output
        timeout: Timeout in seconds (default: 60)

    Returns:
        dict: Verification results with keys:
            - success (bool): Whether verification succeeded
            - env: Lean environment (if successful)
            - errors (list): List of error messages
            - warnings (list): List of warning messages
            - all_messages (list): All messages from Lean
    """
    try:
        if verbose:
            print(f"\nVerifying Lean code (async):\n{lean_code}\n")

        response = await lean_server.async_run(
            Command(cmd=lean_code),
            verbose=verbose,
            timeout=timeout
        )

        messages = response.messages if hasattr(response, 'messages') else []
        errors = [msg for msg in messages if msg.severity == 'error']
        warnings = [msg for msg in messages if msg.severity == 'warning']

        success = len(errors) == 0

        result = {
            'success': success,
            'env': response.env if hasattr(response, 'env') else None,
            'errors': [msg.data for msg in errors],
            'warnings': [msg.data for msg in warnings],
            'all_messages': [{'severity': msg.severity, 'data': msg.data} for msg in messages]
        }

        if verbose:
            print(f"Verification {'succeeded' if success else 'failed'}")
            if errors:
                print(f"Errors: {errors}")

        return result

    except Exception as e:
        return {
            'success': False,
            'env': None,
            'errors': [str(e)],
            'warnings': [],
            'all_messages': []
        }


def create_lean_server(lean_version=None, verbose=False, max_total_memory=0.95):
    """Create and initialize a Lean REPL server.

    Args:
        lean_version: Specific Lean version to use (None for latest)
        verbose: Whether to enable verbose output
        max_total_memory: Max system memory threshold before restart (default 0.95 for macOS)

    Returns:
        AutoLeanServer: Initialized Lean server instance with automatic memory management
    """
    config_kwargs = {'verbose': verbose}
    if lean_version:
        config_kwargs['lean_version'] = lean_version

    config = LeanREPLConfig(**config_kwargs)
    return AutoLeanServer(config, max_total_memory=max_total_memory)

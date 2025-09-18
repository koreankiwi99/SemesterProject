#!/usr/bin/env python3
"""
Simple test of lean-interact functionality
"""
from lean_interact import LeanREPLConfig, LeanServer, Command

def main():
    # Create a Lean REPL configuration
    config = LeanREPLConfig(verbose=True)

    # Start a Lean server with the configuration
    server = LeanServer(config)

    # Execute a simple theorem
    response = server.run(Command(cmd="theorem ex (n : Nat) : n = 5 → n = 5 := id"))

    # Print the response
    print(response)

if __name__ == "__main__":
    main()
#!/usr/bin/env bash
set -e

ssh "$1" "echo ssh_ok && python3 --version"

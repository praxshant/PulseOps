#!/bin/bash
set -e

# Basic ping
curl -f http://localhost:8000/health || exit 1

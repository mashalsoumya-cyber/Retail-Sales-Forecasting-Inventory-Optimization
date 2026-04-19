# 🔧 Troubleshooting Guide

## Installation Issues

### Issue: "pip: command not found"
**Cause:** Python or pip not installed  
**Solution:**
```bash
# Download Python 3.9+ from python.org
# Or use package manager:
# macOS:
brew install python3

# Ubuntu/Debian:
sudo apt-get install python3 python3-pip

# Verify:
python3 --version
pip3 --version
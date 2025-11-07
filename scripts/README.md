# Setup & Cleanup Scripts

## Quick Start

**Install everything:**
```bash
./scripts/setup.sh
```

**Remove everything:**
```bash
./scripts/cleanup.sh
```

---

## What Gets Installed

### 1. Poetry (~50MB)
- **Location**: `~/.local/bin/poetry`
- **Also creates**: `~/.local/share/pypoetry/`
- **Removed by**: Manual uninstall (see below)

### 2. Virtual Environment (~3GB)
- **Location**: `~/.cache/pypoetry/virtualenvs/bacterial-gan-XXXXXX-py3.12/`
- **Contains**: All Python packages including CUDA libraries
- **Removed by**: `./scripts/cleanup.sh` (confirms removal)

### 3. Package Caches (can grow to ~5-10GB)
- **Poetry cache**: `~/.cache/pypoetry/cache/` and `~/.cache/pypoetry/artifacts/`
  - Contains: Downloaded CUDA wheel files (~2.7GB)
- **pip cache**: `~/.cache/pip/`
  - Contains: Additional package caches
- **Removed by**: `./scripts/cleanup.sh` (asks for confirmation)

### 4. Project Files (minimal)
- **Location**: This project directory
- **Contains**:
  - `mlruns/`, `mlartifacts/` (training logs)
  - `models/*.h5` (trained models)
  - Python cache (`__pycache__`, `*.pyc`)
- **Removed by**: `./scripts/cleanup.sh`

---

## What Could Be Left Behind (Residue)

### Potential Residue After cleanup.sh:

1. **Poetry tool itself** (`~/.local/bin/poetry`)
   - **Why**: Shared across all Poetry projects
   - **Remove**: `curl -sSL https://install.python-poetry.org | python3 - --uninstall`

2. **Poetry caches** (if you said "N" to removal prompt)
   - **Location**: `~/.cache/pypoetry/`
   - **Size**: Can be several GB
   - **Remove**: `rm -rf ~/.cache/pypoetry`

3. **pip caches** (if you said "N" to removal prompt)
   - **Location**: `~/.cache/pip/`
   - **Remove**: `rm -rf ~/.cache/pip`

4. **Poetry config**
   - **Location**: `~/.config/pypoetry/`
   - **Size**: Tiny (<1MB)
   - **Remove**: `rm -rf ~/.config/pypoetry`

### Guaranteed NOT to be Removed:

- **System Python**: Scripts don't touch system Python
- **NVIDIA drivers**: Scripts don't touch GPU drivers
- **Other Poetry projects**: Only this project's venv is removed
- **Git repository**: Your `.git/` folder is never touched

---

## Platform Compatibility

### ✅ Works On:
- **Linux** (tested on Ubuntu, Arch)
- **macOS** (should work, untested)
- **Windows WSL** (should work)
- **Windows Git Bash** (should work)

### ❌ Does NOT Work On:
- **Windows CMD/PowerShell** (use WSL instead)

### Known Issues:
1. **PATH not persisting**: After setup, you may need to manually add Poetry to PATH
   ```bash
   # For bash
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

   # For zsh
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
   ```

2. **Different Poetry installation locations**: If Poetry is installed elsewhere (e.g., via apt/brew), scripts may not find it correctly

3. **Cache locations vary**: On some systems, caches might be in different locations:
   - macOS: `~/Library/Caches/pypoetry/`
   - Windows: `%LOCALAPPDATA%\pypoetry\Cache`

---

## Complete Removal Checklist

To remove **absolutely everything**:

```bash
# 1. Run cleanup script
./scripts/cleanup.sh
# Say "y" to all cache removal prompts

# 2. Remove Poetry tool
curl -sSL https://install.python-poetry.org | python3 - --uninstall

# 3. Remove any remaining caches (Linux)
rm -rf ~/.cache/pypoetry
rm -rf ~/.cache/pip
rm -rf ~/.config/pypoetry

# 4. Check for residue
du -sh ~/.cache/pypoetry ~/.cache/pip ~/.config/pypoetry 2>/dev/null
# Should show "No such file or directory" for all

# 5. Remove Poetry from PATH (edit ~/.bashrc or ~/.zshrc)
# Remove the line: export PATH="$HOME/.local/bin:$PATH"

# 6. Delete project (if desired)
cd ..
rm -rf bacterial-gan-augmentation/
```

---

## Verify Disk Space

**Before cleanup:**
```bash
# Check virtual environment size
poetry env info --path | xargs du -sh

# Check cache sizes
du -sh ~/.cache/pypoetry ~/.cache/pip

# Total for this project
du -sh $(poetry env info --path) ~/.cache/pypoetry ~/.cache/pip
```

**After cleanup:**
```bash
# Should show errors or 0 bytes
du -sh ~/.cache/pypoetry ~/.cache/pip 2>/dev/null
```

---

## FAQ

**Q: Why is cleanup.sh asking for confirmation on caches?**
A: Poetry/pip caches are shared across all Python projects. Removing them saves disk space but means re-downloading packages for other projects.

**Q: Will cleanup.sh break my other Poetry projects?**
A: No. It only removes this project's virtual environment. Caches are shared but will be re-downloaded when needed.

**Q: Can I run setup.sh multiple times?**
A: Yes, it's idempotent. It will skip already-installed components.

**Q: What if I want to keep CUDA libraries but remove Python packages?**
A: That's not easily possible since CUDA libraries are installed as Python packages. You'd need to manually manage the cache.

**Q: How do I move the project to another machine?**
A:
1. Copy the project directory (without `mlruns/`, `models/`, etc.)
2. Run `./scripts/setup.sh` on the new machine
3. CUDA libraries will be re-downloaded (~2.7GB)

---

## Troubleshooting

**Problem**: `poetry: command not found` after setup
**Solution**: Add to PATH or use full path: `~/.local/bin/poetry`

**Problem**: Cleanup script says "Poetry not found"
**Solution**: It will still clean project files, just can't remove virtualenv

**Problem**: "Permission denied" errors
**Solution**: Don't use `sudo`. Poetry should be user-installed.

**Problem**: Cleanup doesn't free up space
**Solution**: You probably said "N" to cache removal. Run manually:
```bash
rm -rf ~/.cache/pypoetry ~/.cache/pip
```

**Problem**: Want to remove CUDA but keep other packages
**Solution**: Not possible with these scripts. CUDA is part of TensorFlow installation. You'd need to manually edit the virtualenv, which is not recommended.
